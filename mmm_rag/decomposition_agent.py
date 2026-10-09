"""
decomposition_agent.py
=======================
對應論文 Section II-B「Task Decomposition Agent」。

  1) IsComplex(Q)  -> Eq.(6)：判斷是否為複雜查詢（跨模態推理 / 多步推論 / 意圖模糊）
  2) Decompose(Q)  -> Eq.(7)：若複雜，拆成 n<=3 個子任務
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from PIL import Image

import config
from mmm_rag.model_manager import model_manager

_SYSTEM_PROMPT = (
    "You are the Task Decomposition Agent inside a multimodal financial RAG system. "
    "You must respond with STRICT JSON only, no extra commentary, no markdown fences."
)

_PROMPT_TEMPLATE = """Given the following user question about a financial chart/image, decide:
1. Whether it is a "complex" query, i.e. it requires cross-modal reasoning
   (needs both the image and text), multiple inference steps, or has ambiguous intent.
2. If complex, break it into at most {max_subtasks} short, self-contained sub-questions
   that could each be answered independently by retrieving from a knowledge base.
   If not complex, return an empty list of subtasks.

Question: "{question}"

Respond with JSON exactly in this schema:
{{"is_complex": true/false, "subtasks": ["...", "..."]}}
"""


def _fallback_parse(question: str) -> Dict:
    """LLM 沒有吐出合法 JSON 時的保底邏輯：視為單一、非複雜查詢。"""
    return {"is_complex": False, "subtasks": [question]}


def _extract_json(raw: str) -> Dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


def decompose_query(question: str, query_image: Optional[Image.Image] = None) -> Dict:
    """
    回傳 {"is_complex": bool, "subtasks": List[str], "image_caption": str}。
    當 is_complex=False 時，subtasks 會被正規化成 [question] 這個單一子任務，
    方便呼叫端統一用同一套迴圈處理（對應論文：非複雜查詢直接送去統一檢索）。

    若使用者同時提供了 query_image（例如自己截的一張股價走勢圖），
    文字 LLM 本身看不懂圖片，所以這裡先用 VLM 把圖片內容轉成文字描述，
    再併入拆解 prompt，讓拆出來的子任務能真正對應到這張圖的內容，
    而不是只針對純文字問題做泛泛拆解。
    """
    image_caption = ""
    if query_image is not None:
        image_caption = model_manager.generate_vlm(
            query_image,
            "請詳細描述這張財經/股票走勢圖的內容：股票代碼或標的（若可辨識）、"
            "時間區間、價格走勢型態、成交量變化、以及與大盤或其他指標的相對表現等關鍵資訊。",
            max_new_tokens=200,
        )
        print(f"  [DecompositionAgent] 查詢圖片描述: {image_caption}")

    prompt = _PROMPT_TEMPLATE.format(max_subtasks=config.MAX_SUBTASKS, question=question)
    if image_caption:
        prompt += (
            "\n使用者同時提供了一張圖片，以下是 VLM 對該圖片內容的描述，"
            f"請將其視為問題的一部分，拆解子任務時也要考慮這張圖的內容：\n{image_caption}\n"
        )

    raw = model_manager.generate_text(
        prompt, system=_SYSTEM_PROMPT, max_new_tokens=config.MAX_NEW_TOKENS_SHORT
    )

    try:
        parsed = _extract_json(raw)
        is_complex = bool(parsed.get("is_complex", False))
        subtasks = parsed.get("subtasks", []) or []
        subtasks = [s.strip() for s in subtasks if isinstance(s, str) and s.strip()]
        subtasks = subtasks[: config.MAX_SUBTASKS]
    except Exception as e:  # noqa: BLE001
        print(f"[DecompositionAgent] JSON 解析失敗（{e}），使用保底邏輯。原始輸出：{raw!r}")
        parsed = _fallback_parse(question)
        is_complex, subtasks = parsed["is_complex"], parsed["subtasks"]

    if not is_complex or not subtasks:
        is_complex = False
        subtasks = [question]

    return {"is_complex": is_complex, "subtasks": subtasks, "image_caption": image_caption}
