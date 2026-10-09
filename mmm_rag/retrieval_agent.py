"""
retrieval_agent.py
===================
對應論文 Section II-C「Multi-Feature Multi-Source Retrieval Agent」。

三個來源、三個子答案：
  - A_visual   <- Eq.(8)(9)(10)：CLIP 文字塔編碼子任務 -> 查 Visual Feature DB -> VLM 生成答案
  - A_semantic <- Eq.(11)(12)(13)(14)：BGE 稠密檢索 + Graph 檢索 -> LLM 融合生成答案
  - A_web      <- Eq.(15)：DuckDuckGo (ddgs) 檢索 -> LLM 摘要生成答案
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image
import numpy as np
import config
from mmm_rag.graph_store import GraphStore
from mmm_rag.model_manager import model_manager
from mmm_rag.vector_store import FaissStore


# ----------------------------------------------------------------------
# 1) Visual-Feature Knowledge Base Retrieval
# ----------------------------------------------------------------------
def retrieve_visual(
    subtask: str,
    visual_store: FaissStore,
    query_image: Image.Image = None,
    top_k: int = None,
) -> Dict:
    """對應 Eq.(8)-(10)。回傳 {"answer": str, "refs": [...]}.

    query_image 是使用者自己提供的查詢圖片（例如自己截的股價走勢圖），可選。
    """
    top_k = top_k or config.TOP_K
    text_vec = model_manager.encode_text_clip(subtask)

    # 建庫時每筆存的是 F = Concat(v_I, v_T)，是 512+512=1024 維。
    # 若使用者有提供查詢圖片，直接用它真正的 CLIP 圖片向量去對齊 v_I 的位置，
    # 這樣檢索出來的會是「長得像使用者這張圖」的歷史圖表，更準確；
    # 若沒有提供查詢圖片，才退回用文字向量複製一份頂替（維度對齊，但語意較弱）。
    if query_image is not None:
        image_vec = model_manager.encode_image_clip(query_image)
    else:
        image_vec = text_vec
    query_vec = np.concatenate([image_vec, text_vec], axis=0)
    hits = visual_store.search(query_vec, top_k=top_k)

    if not hits:
        return {"answer": "(視覺特徵資料庫中沒有找到相關資料)", "refs": []}

    # 取分數最高的一筆圖文對，丟給 VLM 生成答案（如需更嚴謹可以對每筆都生成再挑最好的）
    score, meta = hits[0]
    image_path = meta["image_path"]
    context_text = meta.get("text", "")

    try:
        retrieved_image = Image.open(image_path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return {"answer": f"(圖片讀取失敗: {e})", "refs": []}

    if query_image is not None:
        # 同時把「使用者查詢圖片」與「資料庫檢索到的參考圖片」給 VLM，讓它做比較
        prompt = (
            "You are analyzing financial chart images. The FIRST image is the user's own "
            "query chart. The SECOND image is the most similar reference chart retrieved "
            "from the knowledge base, with the following associated caption/context:\n"
            f"{context_text}\n\n"
            f"Question: {subtask}\n"
            "Compare the two charts' trends and answer concisely, using the reference "
            "chart's context to support your reasoning about the query chart."
        )
        answer = model_manager.generate_vlm([query_image, retrieved_image], prompt)
    else:
        prompt = (
            "You are analyzing a financial chart image. Here is some associated caption/context:\n"
            f"{context_text}\n\n"
            f"Question: {subtask}\n"
            "Answer concisely based on what you can see in the chart and the context above."
        )
        answer = model_manager.generate_vlm(retrieved_image, prompt)

    refs = [{"doc_id": m.get("doc_id"), "image_path": m.get("image_path"), "score": s} for s, m in hits]
    return {"answer": answer, "refs": refs}


# ----------------------------------------------------------------------
# 2) Semantic-Feature Knowledge Base Retrieval (dense + graph)
# ----------------------------------------------------------------------
def retrieve_semantic(
    subtask: str,
    semantic_store: FaissStore,
    graph_store: GraphStore,
    top_k: int = None,
    graph_depth: int = None,
) -> Dict:
    """對應 Eq.(11)-(14)。回傳 {"answer": str, "refs": [...]}."""
    top_k = top_k or config.TOP_K
    graph_depth = graph_depth if graph_depth is not None else config.GRAPH_TRAVERSAL_DEPTH

    # a) Dense-vector retrieval  (Eq.11, 12)
    query_vec = model_manager.encode_text_semantic([subtask])[0]
    dense_hits = semantic_store.search(query_vec, top_k=top_k)

    # b) Graph-based retrieval (Eq.13)
    graph_doc_ids = set(graph_store.query(subtask, depth=graph_depth, max_docs=top_k))
    doc_id_to_text = {m["doc_id"]: m["text"] for m in semantic_store.metadata}
    graph_hits_text = [doc_id_to_text[d] for d in graph_doc_ids if d in doc_id_to_text]

    dense_hits_text = [m["text"] for _, m in dense_hits]

    combined_docs = dense_hits_text + [t for t in graph_hits_text if t not in dense_hits_text]
    if not combined_docs:
        return {"answer": "(語義特徵資料庫中沒有找到相關資料)", "refs": []}

    context = "\n---\n".join(combined_docs[: max(top_k, 1) * 2])
    prompt = (
        "Use the following retrieved financial documents (from dense vector search "
        "and knowledge-graph traversal) to answer the question. Be concise.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {subtask}\nAnswer:"
    )
    answer = model_manager.generate_text(prompt, max_new_tokens=config.MAX_NEW_TOKENS_SHORT)

    refs = [{"doc_id": m.get("doc_id"), "score": s} for s, m in dense_hits] + [
        {"doc_id": d, "score": None, "source": "graph"} for d in graph_doc_ids
    ]
    return {"answer": answer, "refs": refs}


# ----------------------------------------------------------------------
# 3) Web Retrieval
# ----------------------------------------------------------------------
def retrieve_web(subtask: str, max_results: int = None) -> Dict:
    """對應 Eq.(15)：用 ddgs（原 duckduckgo_search）查詢並摘要。"""
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(subtask, max_results=max_results))
    except Exception as e:  # noqa: BLE001
        return {"answer": f"(網路檢索失敗: {e})", "refs": []}

    if not results:
        return {"answer": "(網路上沒有找到相關資料)", "refs": []}

    snippets = "\n---\n".join(f"{r.get('title', '')}: {r.get('body', '')}" for r in results)
    prompt = (
        "Summarize the following web search snippets into a concise answer "
        f"to the question below.\n\nSnippets:\n{snippets}\n\nQuestion: {subtask}\nAnswer:"
    )
    answer = model_manager.generate_text(prompt, max_new_tokens=config.MAX_NEW_TOKENS_SHORT)

    refs = [{"title": r.get("title"), "url": r.get("href")} for r in results]
    return {"answer": answer, "refs": refs}


# ----------------------------------------------------------------------
# 整合三個來源
# ----------------------------------------------------------------------
def retrieve_all_sources(
    subtask: str,
    semantic_store: FaissStore,
    visual_store: FaissStore,
    graph_store: GraphStore,
    query_image: Image.Image = None,
) -> Dict[str, Dict]:
    """回傳論文中的三個候選答案 {visual, semantic, web}。

    query_image 只影響 visual 檢索（真實圖片比對），semantic / web 仍走純文字。
    """
    visual = retrieve_visual(subtask, visual_store, query_image=query_image)
    semantic = retrieve_semantic(subtask, semantic_store, graph_store)
    web = retrieve_web(subtask)
    return {"visual": visual, "semantic": semantic, "web": web}
