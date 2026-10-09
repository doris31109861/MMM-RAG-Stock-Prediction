"""
03_build_knowledge_base.py
===========================
對應論文 Section II-A「Construction of Multi-Feature Databases」，
也就是 Fig.1(b) / Fig.2 左上角的 Feature-Parsing 流程。

對 data/metadata.jsonl 裡的每一筆 (I_db, T_db)：

  Semantic feature DB:
    D(I) = VLM(I)                         -- Eq.(1)，用 Qwen2-VL 生成圖片描述
    S    = Concat(T_db, D(I))             -- Eq.(2)
    S 用 BGE 做 embedding，存進 semantic_feature_db.faiss

  Visual feature DB:
    v_I = CLIP_image(I)                   -- Eq.(3)（原論文用 BLIP-2 ViT，這裡用 CLIP 對應）
    v_T = CLIP_text(T_db)                 -- Eq.(4)
    F   = Concat(v_I, v_T)                -- Eq.(5)
    F 存進 visual_feature_db.faiss

用法：
    python scripts/03_build_knowledge_base.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from mmm_rag.model_manager import model_manager  # noqa: E402
from mmm_rag.vector_store import FaissStore  # noqa: E402

_CAPTION_PROMPT = (
    "Describe this financial chart in 2-3 sentences. Mention the type of chart, "
    "what metric/quantity it shows, and any notable trend, if visible."
)


def load_metadata():
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {config.METADATA_PATH}，請先執行 python scripts/02_download_dataset.py"
        )
    records = []
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    records = load_metadata()
    print(f"讀取到 {len(records)} 筆知識庫原始資料 (I_db, T_db)。")

    semantic_texts = []
    semantic_meta = []
    visual_vectors = []
    visual_meta = []

    for rec in tqdm(records, desc="Feature-Parsing Agent 建置資料庫中"):
        image_path = rec["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  跳過 {rec['doc_id']}，圖片開啟失敗: {e}")
            continue

        T_db = f"{rec.get('verified_caption', '')}. {rec.get('related_sentences', '')}".strip()

        # ---- Semantic Feature Database ----
        # Eq.(1): D(I) = Qwen2.5-VL(I)  -> 這裡用 Qwen2-VL-2B-Instruct 當本地替代品
        try:
            D_I = model_manager.generate_vlm(image, _CAPTION_PROMPT, max_new_tokens=128)
        except Exception as e:  # noqa: BLE001
            print(f"  {rec['doc_id']} VLM 生成描述失敗，改用空字串: {e}")
            D_I = ""

        # Eq.(2): S = Concat(T, D(I))
        S = f"{T_db}\n[VLM description] {D_I}".strip()
        semantic_texts.append(S)
        semantic_meta.append(
            {
                "doc_id": rec["doc_id"],
                "text": S,
                "image_path": image_path,
                "raw_T": T_db,
                "vlm_caption": D_I,
            }
        )

        # ---- Visual Feature Database ----
        # Eq.(3): v_I = ViT(I)   (這裡用 CLIP image tower)
        # Eq.(4): v_T = TextEncoder(T)  (這裡用 CLIP text tower)
        v_I = model_manager.encode_image_clip(image)
        v_T = model_manager.encode_text_clip(T_db[:300] if T_db else " ")
        # Eq.(5): F = Concat(v_I, v_T)
        F = np.concatenate([v_I, v_T], axis=0)
        visual_vectors.append(F)
        visual_meta.append({"doc_id": rec["doc_id"], "image_path": image_path, "text": T_db})

    if not semantic_texts:
        print("沒有任何成功處理的資料，請檢查 data/images 與 metadata。")
        return

    print("\n用 BGE 產生 Semantic Feature Database 的向量 (Eq.11) ...")
    semantic_vectors = model_manager.encode_text_semantic(semantic_texts)
    semantic_store = FaissStore(dim=semantic_vectors.shape[1])
    semantic_store.add(semantic_vectors, semantic_meta)
    semantic_store.save(config.SEMANTIC_INDEX_PATH, config.SEMANTIC_META_PATH)
    print(f"  Semantic Feature DB 已儲存: {config.SEMANTIC_INDEX_PATH}  (共 {len(semantic_store)} 筆)")

    print("\n儲存 Visual Feature Database ...")
    visual_vectors = np.stack(visual_vectors, axis=0)
    visual_store = FaissStore(dim=visual_vectors.shape[1])
    visual_store.add(visual_vectors, visual_meta)
    visual_store.save(config.VISUAL_INDEX_PATH, config.VISUAL_META_PATH)
    print(f"  Visual Feature DB 已儲存:   {config.VISUAL_INDEX_PATH}  (共 {len(visual_store)} 筆)")

    print("\n完成！接下來執行: python scripts/04_build_graph_index.py")


if __name__ == "__main__":
    main()
