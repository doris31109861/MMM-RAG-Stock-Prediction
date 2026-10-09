"""
main.py
=======
整個 MMM-RAG（財經/股市版）流程的入口，對應論文 Fig.2 的三個 Agent 依序執行：

    Task Decomposition Agent
            |
            v
    Multi-Feature Multi-Source Retrieval Agent (visual / semantic / web)
            |
            v
    Consensus Voting and Answer Generation Agent

前置作業（第一次使用前務必依序跑過）：
    python scripts/01_setup_check.py
    python scripts/02_download_dataset.py
    python scripts/03_build_knowledge_base.py
    python scripts/04_build_graph_index.py

用法範例：
    python main.py --question "根據這批財經圖表資料庫，投資人在解讀庫存/營收類股價圖時，
                                通常會觀察哪些跨期趨勢訊號？"

    # 或者直接從 FinMME 抽一筆現成的問題來測試整個流程：
    python main.py --sample
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from mmm_rag import decision_agent, decomposition_agent, retrieval_agent  # noqa: E402
from mmm_rag.graph_store import GraphStore  # noqa: E402
from mmm_rag.vector_store import FaissStore  # noqa: E402


def load_stores():
    for p in (config.SEMANTIC_INDEX_PATH, config.VISUAL_INDEX_PATH, config.GRAPH_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"找不到 {p}。請先依序執行 scripts/02, 03, 04 建立資料庫，詳見 README.md。"
            )
    semantic_store = FaissStore.load(config.SEMANTIC_INDEX_PATH, config.SEMANTIC_META_PATH)
    visual_store = FaissStore.load(config.VISUAL_INDEX_PATH, config.VISUAL_META_PATH)
    graph_store = GraphStore.load(config.GRAPH_PATH)
    return semantic_store, visual_store, graph_store


def pick_sample_question() -> str:
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError("找不到 metadata.jsonl，請先執行 scripts/02_download_dataset.py")
    records = []
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    rec = random.choice(records)
    print(f"[--sample] 從 FinMME 抽到一筆問題 (doc_id={rec['doc_id']}):")
    print(f"           {rec['question_text']}")
    if rec.get("options"):
        print(f"           選項: {rec['options']}")
    print(f"           (資料集標準答案僅供參考: {rec.get('answer')})\n")
    return rec["question_text"]


def run_subtask(subtask: str, semantic_store, visual_store, graph_store, query_image=None) -> dict:
    print(f"\n  >>> 處理子任務: {subtask}")
    t0 = time.time()
    candidates = retrieval_agent.retrieve_all_sources(
        subtask, semantic_store, visual_store, graph_store, query_image=query_image
    )
    for source, result in candidates.items():
        preview = result["answer"][:120].replace("\n", " ")
        print(f"      [{source:8s}] {preview}")
    decision = decision_agent.consensus_vote_and_generate(subtask, candidates)
    print(f"      -> 共識權重: {decision['weights']}")
    print(f"      -> 子任務答案: {decision['final_answer'][:200]}")
    print(f"      (耗時 {time.time() - t0:.1f}s)")
    return {"subtask": subtask, **decision}


_SYNTHESIS_PROMPT = """The original user question was decomposed into sub-questions,
each of which was answered separately. Combine these sub-answers into one final,
coherent answer to the ORIGINAL question.

Original question: {question}

{subtask_block}

Final combined answer:"""


def synthesize_final_answer(question: str, subtask_results: list) -> str:
    if len(subtask_results) == 1:
        return subtask_results[0]["final_answer"]

    from mmm_rag.model_manager import model_manager

    block = "\n".join(
        f"- Sub-question: {r['subtask']}\n  Sub-answer: {r['final_answer']}" for r in subtask_results
    )
    prompt = _SYNTHESIS_PROMPT.format(question=question, subtask_block=block)
    return model_manager.generate_text(prompt, max_new_tokens=config.MAX_NEW_TOKENS_LONG)


def main():
    parser = argparse.ArgumentParser(description="Run the MMM-RAG (finance edition) pipeline on one question.")
    parser.add_argument("--question", type=str, default=None, help="要詢問系統的財經/股市相關問題")
    parser.add_argument("--sample", action="store_true", help="從 FinMME metadata 隨機抽一筆問題來測試")
    parser.add_argument(
        "--image", type=str, default=None,
        help="使用者自己提供的查詢圖片路徑（可選，例如自己截的股價走勢圖）",
    )
    parser.add_argument("--save-log", action="store_true", default=True, help="是否把完整過程存成 JSON log")
    args = parser.parse_args()

    if args.sample:
        question = pick_sample_question()
    elif args.question:
        question = args.question
    else:
        parser.error("請用 --question \"你的問題\" 或 --sample 其中一種方式提供問題。")
        return

    query_image = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            parser.error(f"找不到圖片: {image_path}")
        try:
            query_image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            parser.error(f"圖片讀取失敗: {e}")

    print("=" * 70)
    print("MMM-RAG (財經/股市版) 開始執行")
    print("=" * 70)
    print(f"原始問題: {question}")
    if query_image is not None:
        print(f"查詢圖片: {args.image}")
    print()

    semantic_store, visual_store, graph_store = load_stores()
    print(f"已載入資料庫: semantic={len(semantic_store)} 筆, visual={len(visual_store)} 筆, "
          f"graph={graph_store.describe()}\n")

    # ---- Step 1: Task Decomposition Agent ----
    print("[Step 1] Task Decomposition Agent ...")
    decomposition = decomposition_agent.decompose_query(question, query_image=query_image)
    print(f"  is_complex = {decomposition['is_complex']}")
    print(f"  subtasks   = {decomposition['subtasks']}")

    # ---- Step 2 & 3: Retrieval + Consensus Voting，逐個子任務處理 ----
    print("\n[Step 2+3] Multi-Source Retrieval + Consensus Voting ...")
    subtask_results = [
        run_subtask(st, semantic_store, visual_store, graph_store, query_image=query_image)
        for st in decomposition["subtasks"]
    ]

    # ---- 若有多個子任務，再融合成對原始問題的最終回答 ----
    final_answer = synthesize_final_answer(question, subtask_results)

    print("\n" + "=" * 70)
    print("最終答案")
    print("=" * 70)
    print(final_answer)
    print("=" * 70)

    if args.save_log:
        log = {
            "question": question,
            "query_image": args.image,
            "decomposition": decomposition,
            "subtask_results": subtask_results,
            "final_answer": final_answer,
        }
        log_path = config.LOG_DIR / f"run_{int(time.time())}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"\n完整過程已存到: {log_path}")


if __name__ == "__main__":
    main()
