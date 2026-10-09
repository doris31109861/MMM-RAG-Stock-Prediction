"""
05_evaluate.py
==============
在 FinMME 上評估 MMM-RAG 的準確率，並做消融實驗（ablation）比較各元件的貢獻。

為了避免「題目的圖片本身就在知識庫裡」造成資料洩漏，評估題目取知識庫之後的資料：
知識庫是 FinMME 的前 DATASET_SUBSET_SIZE 筆，評估題目從第 DATASET_SUBSET_SIZE 筆開始取 --n 筆，
並把題目附的圖表當作查詢圖片（--image）丟進系統。

模式（--mode，可給多個）：
  full      完整 MMM-RAG：任務拆解 → visual / semantic / web 三路檢索 → 共識投票（論文方法）
  visual    只用視覺特徵庫檢索 → VLM 作答
  semantic  只用語義特徵庫＋知識圖譜檢索 → LLM 作答
  web       只用網路檢索 → LLM 作答
  vlm_only  不做任何檢索，VLM 直接看圖作答（baseline）

用法（需要 GPU，並先跑過 scripts/02–04）：
    python scripts/05_evaluate.py --n 100 --mode vlm_only visual semantic full
結果：results/eval_<mode>.jsonl（逐題紀錄）、results/summary.md（各模式準確率比較表）
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from mmm_rag.evaluation import format_question, is_correct, summarize  # noqa: E402

RESULTS_DIR = config.ROOT_DIR / "results"
EVAL_IMAGE_DIR = config.DATA_DIR / "eval_images"
MODES = ["full", "visual", "semantic", "web", "vlm_only"]


def load_eval_records(n: int):
    """取知識庫之後的 n 筆 FinMME 資料作為評估題目，圖片存到 data/eval_images/。"""
    from datasets import load_dataset

    EVAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(config.HF_DATASET_NAME, split=config.HF_DATASET_SPLIT)
    start = config.DATASET_SUBSET_SIZE or 0
    if config.DATASET_SUBSET_SIZE is None:
        print("注意：知識庫是全量資料，評估題目會和知識庫重疊（有資料洩漏）。")
    end = min(start + n, len(ds))
    records = []
    for row in ds.select(range(start, end)):
        path = EVAL_IMAGE_DIR / f"{row['id']}.jpg"
        if not path.exists():
            row["image"].convert("RGB").save(path, format="JPEG", quality=90)
        records.append({k: row.get(k) for k in
                        ("id", "question_text", "question_type", "options", "answer", "unit", "tolerance")}
                       | {"image_path": str(path)})
    return records


def answer_question(mode: str, question: str, image, stores):
    """依模式回傳系統的作答文字。"""
    from mmm_rag import decomposition_agent, retrieval_agent
    from mmm_rag.model_manager import model_manager
    import main as pipeline

    semantic_store, visual_store, graph_store = stores
    if mode == "vlm_only":
        return model_manager.generate_vlm(image, question)
    if mode == "visual":
        return retrieval_agent.retrieve_visual(question, visual_store, query_image=image)["answer"]
    if mode == "semantic":
        return retrieval_agent.retrieve_semantic(question, semantic_store, graph_store)["answer"]
    if mode == "web":
        return retrieval_agent.retrieve_web(question)["answer"]
    # full：與 main.py 相同的完整流程
    decomposition = decomposition_agent.decompose_query(question, query_image=image)
    results = [pipeline.run_subtask(st, semantic_store, visual_store, graph_store, query_image=image)
               for st in decomposition["subtasks"]]
    return pipeline.synthesize_final_answer(question, results)


def write_summary(all_summaries: dict, n: int):
    """把各模式的準確率整理成 Markdown 表格。"""
    types = sorted({t for s in all_summaries.values() for t in s["by_type"]})
    lines = [f"# FinMME 評估結果（{n} 題，知識庫之後的資料）", "",
             "| 模式 | 整體 | " + " | ".join(types) + " |",
             "|---|---|" + "---|" * len(types)]
    for mode, s in all_summaries.items():
        cells = [f"{s['by_type'].get(t, {}).get('accuracy', '-')} (n={s['by_type'].get(t, {}).get('n', 0)})"
                 for t in types]
        lines.append(f"| {mode} | {s['accuracy']} | " + " | ".join(cells) + " |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMM-RAG on held-out FinMME questions.")
    parser.add_argument("--n", type=int, default=50, help="評估題數（取知識庫之後的資料）")
    parser.add_argument("--mode", nargs="+", default=["vlm_only", "full"], choices=MODES)
    args = parser.parse_args()

    from PIL import Image
    import main as pipeline

    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_eval_records(args.n)
    stores = pipeline.load_stores()
    print(f"評估 {len(records)} 題，模式：{args.mode}")

    all_summaries = {}
    for mode in args.mode:
        out_path = RESULTS_DIR / f"eval_{mode}.jsonl"
        results = []
        with open(out_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(records, 1):
                question = format_question(rec)
                image = Image.open(rec["image_path"]).convert("RGB")
                t0 = time.time()
                try:
                    prediction = answer_question(mode, question, image, stores)
                except Exception as e:  # noqa: BLE001  單題失敗記錄下來，不中斷整個評估
                    prediction = f"(error: {e})"
                row = {"id": rec["id"], "question_type": rec["question_type"], "answer": rec["answer"],
                       "prediction": prediction, "correct": is_correct(rec, prediction),
                       "seconds": round(time.time() - t0, 1)}
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{mode}] {i}/{len(records)} id={rec['id']} correct={row['correct']} ({row['seconds']}s)")
        all_summaries[mode] = summarize(results)
        print(f"[{mode}] accuracy = {all_summaries[mode]['accuracy']}")

    write_summary(all_summaries, len(records))


if __name__ == "__main__":
    main()
