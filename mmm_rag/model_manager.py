"""
model_manager.py
=================
統一管理四個模型的載入 / 釋放：

  - text_llm : 文字 LLM  -> Task Decomposition / Semantic&Web 子答案 / Consensus Voting / 最終整合
  - vlm      : 視覺語言模型 -> 建庫時生成圖片描述 D(I)、查詢時的 Visual 子答案
  - clip     : 影像-文字雙塔編碼 -> Visual Feature Database
  - embedder : 文字 Embedding -> Semantic Feature Database 的稠密檢索

在 8GB VRAM 的顯卡（例如 RTX 5060 8GB）上，重點是「4-bit 量化 + 視需要卸載」。
把所有跟顯卡資源管理相關的邏輯都集中在這支檔案，
其餘 agent 程式完全不用煩惱 VRAM 問題，呼叫 ModelManager 的方法即可。
"""

from __future__ import annotations

import gc
from typing import List, Optional

import torch
from PIL import Image

import config


def _bnb_config():
    """建立 4-bit 量化設定 (bitsandbytes NF4)。"""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class ModelManager:
    """單例：整個程式共用同一個 ModelManager，避免重複載入模型。"""

    _instance: Optional["ModelManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = config.DEVICE if torch.cuda.is_available() else "cpu"
        if config.DEVICE == "cuda" and not torch.cuda.is_available():
            print("[ModelManager] 警告：偵測不到 CUDA，改用 CPU（速度會非常慢）。")

        self._llm = None
        self._llm_tok = None
        self._vlm = None
        self._vlm_proc = None
        self._clip = None
        self._clip_proc = None
        self._embedder = None

    # ------------------------------------------------------------------
    # 文字 LLM
    # ------------------------------------------------------------------
    def get_llm(self):
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[ModelManager] 載入文字 LLM: {config.TEXT_LLM_MODEL} ...")
            self._llm_tok = AutoTokenizer.from_pretrained(config.TEXT_LLM_MODEL)
            kwargs = dict(device_map="auto" if self.device == "cuda" else None)
            if config.USE_4BIT and self.device == "cuda":
                kwargs["quantization_config"] = _bnb_config()
            else:
                kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            self._llm = AutoModelForCausalLM.from_pretrained(config.TEXT_LLM_MODEL, **kwargs)
            self._llm.eval()
        return self._llm, self._llm_tok

    def unload_llm(self):
        if self._llm is not None:
            del self._llm
            del self._llm_tok
            self._llm, self._llm_tok = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = None,
    ) -> str:
        """用文字 LLM 依聊天模板生成回覆（純文字，無圖片）。"""
        model, tok = self.get_llm()
        max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS_SHORT

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        result = tok.decode(gen_ids, skip_special_tokens=True).strip()

        if config.LOW_VRAM_MODE:
            self.unload_llm()
        return result

    # ------------------------------------------------------------------
    # 視覺語言模型 VLM (Qwen2-VL)
    # ------------------------------------------------------------------
    def get_vlm(self):
        if self._vlm is None:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            print(f"[ModelManager] 載入 VLM: {config.VLM_MODEL} ...")
            self._vlm_proc = AutoProcessor.from_pretrained(config.VLM_MODEL)
            kwargs = dict(device_map="auto" if self.device == "cuda" else None)
            if config.USE_4BIT and self.device == "cuda":
                kwargs["quantization_config"] = _bnb_config()
            else:
                kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            self._vlm = Qwen2VLForConditionalGeneration.from_pretrained(config.VLM_MODEL, **kwargs)
            self._vlm.eval()
        return self._vlm, self._vlm_proc

    def unload_vlm(self):
        if self._vlm is not None:
            del self._vlm
            del self._vlm_proc
            self._vlm, self._vlm_proc = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_vlm(self, image, prompt: str, max_new_tokens: int = None) -> str:
        """
        VLM(I, prompt) -> 文字。

        image 可以是單一 PIL.Image，也可以是 List[PIL.Image]（例如同時傳入
        「使用者查詢圖片」+「資料庫檢索到的參考圖片」兩張圖給模型比較）。
        Qwen2-VL 原生支援多圖輸入，這裡只是把 content 依圖片數量展開。

        對應論文 Eq.(1) D(I) = Qwen2.5-VL(I) 以及 Eq.(10) A_visual = VLM(I_j, T_j)。
        """
        model, processor = self.get_vlm()
        max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS_SHORT

        images = image if isinstance(image, list) else [image]

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if config.LOW_VRAM_MODE:
            self.unload_vlm()
        return result

    # ------------------------------------------------------------------
    # CLIP（Visual Feature Database 用）
    # ------------------------------------------------------------------
    def get_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            print(f"[ModelManager] 載入 CLIP: {config.CLIP_MODEL} ...")
            self._clip = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
            self._clip.eval()
            self._clip_proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL, use_fast=True)
        return self._clip, self._clip_proc

    @staticmethod
    def _pooled(output):
        """
        相容不同 transformers 版本：舊版 vision_model/text_model 回傳 tuple，
        新版回傳 BaseModelOutputWithPooling（有 .pooler_output）。
        """
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output[1]

    def encode_image_clip(self, image: Image.Image):
        """
        對應論文 Eq.(3): v_I = ViT_BLIP-2(I)，這裡改用 CLIP 的 image tower。

        注意：不直接呼叫 model.get_image_features()，因為 transformers 5.x 之後
        這個便利函式的回傳型別改成 BaseModelOutputWithPooling（而不是 tensor），
        對 `.norm()` 這種 tensor 方法會直接拋 AttributeError。改成手動呼叫
        vision_model + visual_projection，這兩個子模組的介面在各版本間穩定得多，
        即使 requirements.txt 已經把 transformers 釘在 <5.0，這裡多一層防護也無妨。
        """
        model, proc = self.get_clip()
        inputs = proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = self._pooled(vision_out)
            feat = model.visual_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def encode_text_clip(self, text: str):
        """對應論文 Eq.(4): v_T = TextEncoder_BLIP-2(T)，這裡改用 CLIP 的 text tower。
        原因同 encode_image_clip：改用 text_model + text_projection 手動組合，
        避開 get_text_features() 在新版 transformers 的回傳型別變動。
        """
        model, proc = self.get_clip()
        inputs = proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(
            self.device
        )
        with torch.no_grad():
            text_out = model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
            pooled = self._pooled(text_out)
            feat = model.text_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # 文字 Embedding（Semantic Feature Database 用）
    # ------------------------------------------------------------------
    def get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            print(f"[ModelManager] 載入文字 Embedding 模型: {config.EMBED_MODEL} ...")
            self._embedder = SentenceTransformer(config.EMBED_MODEL, device=self.device)
        return self._embedder

    def encode_text_semantic(self, texts: List[str]):
        """對應論文 Eq.(11): e_Ti = Embed(Ti; BGE)。"""
        embedder = self.get_embedder()
        return embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


# 全域共用單例
model_manager = ModelManager()
