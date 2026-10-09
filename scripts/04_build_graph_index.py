"""
04_build_graph_index.py
=========================
對應論文 Section II-C-2-b「Graph-based retrieval」，用 mmm_rag/graph_store.py
裡簡化版的 GraphStore，從 Semantic Feature Database 的文字內容建出一個
entity <-> document 的二分圖 G=(V,E)，供查詢時做 GraphTraversal。

依賴 03_build_knowledge_base.py 產生的 semantic_feature_db.meta.pkl，
所以請先跑過 03 再跑這支。

用法：
    python scripts/04_build_graph_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from mmm_rag.graph_store import GraphStore  # noqa: E402
from mmm_rag.vector_store import FaissStore  # noqa: E402


def main():
    if not config.SEMANTIC_META_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {config.SEMANTIC_META_PATH}，請先執行 python scripts/03_build_knowledge_base.py"
        )

    semantic_store = FaissStore.load(config.SEMANTIC_INDEX_PATH, config.SEMANTIC_META_PATH)
    docs = [{"doc_id": m["doc_id"], "text": m["text"]} for m in semantic_store.metadata]
    print(f"讀取到 {len(docs)} 份語義文件，開始用 LlamaIndex 建立知識圖譜 ...")
    print("（會對每個 chunk 呼叫一次 LLM 做三元組抽取，資料筆數多時請耐心等待）\n")

    graph_store = GraphStore()
    graph_store.build_from_docs(docs)
    graph_store.save(config.GRAPH_PATH)

    print(f"\n完成！知識圖譜已儲存到: {config.GRAPH_PATH}")
    print("（LlamaIndex 會在這個資料夾底下存多個 json 檔，可以打開")
    print(" graph_store.json 檢視實際抽取出的三元組內容）")
    print("\n三個資料庫都已就緒！接下來可以執行: python main.py --question \"...\"")


if __name__ == "__main__":
    main()
