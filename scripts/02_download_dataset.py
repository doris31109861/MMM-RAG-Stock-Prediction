"""
02_download_dataset.py
=======================
從 HuggingFace 下載 luojunyu/FinMME 資料集，
把它轉成 MMM-RAG 需要的 (I_db, T_db) 原始知識庫格式：

    I_db : 圖片本身（財經圖表），存成 data/images/{id}.jpg
    T_db : 與圖片關聯的原始文字說明
           （這裡用 verified_caption + related_sentences 組成，
             對應論文中「與影像關聯的原始文本說明」）

另外把 FinMME 原本的 question_text / options / answer 等欄位也保留下來，
可以在 05 的查詢階段當作現成的測試問題來源（雖然本專案主要示範架構，
你也可以完全用自己的財經 / 股市問題來問）。

用法：
    python scripts/02_download_dataset.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402


def main():
    print(f"下載資料集: {config.HF_DATASET_NAME} (split={config.HF_DATASET_SPLIT}) ...")
    print("第一次執行會從 HuggingFace Hub 下載並快取到 HF_HOME，之後重跑會直接讀取快取。")
    dataset = load_dataset(config.HF_DATASET_NAME, split=config.HF_DATASET_SPLIT)

    if config.DATASET_SUBSET_SIZE is not None:
        n = min(config.DATASET_SUBSET_SIZE, len(dataset))
        print(f"只取前 {n} 筆做 demo（config.DATASET_SUBSET_SIZE={config.DATASET_SUBSET_SIZE}）。"
              f" 全量共 {len(dataset)} 筆，想跑全量請把 config.py 的 DATASET_SUBSET_SIZE 設成 None。")
        dataset = dataset.select(range(n))
    else:
        print(f"將處理全量 {len(dataset)} 筆資料，這會花較長時間。")

    records = []
    for row in tqdm(dataset, desc="轉存圖片與整理 metadata"):
        doc_id = row["id"]
        image = row["image"]  # PIL.Image
        image_path = config.IMAGE_DIR / f"{doc_id}.jpg"
        try:
            image.convert("RGB").save(image_path, format="JPEG", quality=90)
        except Exception as e:  # noqa: BLE001
            print(f"  跳過 id={doc_id}，圖片存檔失敗: {e}")
            continue

        record = {
            "doc_id": doc_id,
            "image_path": str(image_path),
            "verified_caption": row.get("verified_caption", "") or "",
            "related_sentences": row.get("related_sentences", "") or "",
            # 以下欄位保留供 05_run_query.py 選用作為測試問題
            "question_text": row.get("question_text", ""),
            "question_type": row.get("question_type", ""),
            "options": row.get("options", ""),
            "answer": row.get("answer", ""),
            "unit": row.get("unit", ""),
        }
        records.append(record)

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成！共處理 {len(records)} 筆資料。")
    print(f"  圖片存放於: {config.IMAGE_DIR}")
    print(f"  metadata:   {config.METADATA_PATH}")
    print("接下來執行: python scripts/03_build_knowledge_base.py")


if __name__ == "__main__":
    main()
