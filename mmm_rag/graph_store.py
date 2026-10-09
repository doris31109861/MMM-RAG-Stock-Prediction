"""
graph_store.py
===============
對應論文 Section II-C-2-b「Graph-based retrieval」，原論文用 LlamaIndex
建立 G=(V,E) 再做 GraphTraversal(G, e_Ti, d)。

這裡改用 LlamaIndex 的 KnowledgeGraphIndex 忠實重現原論文的做法：

  - 建庫（build_from_docs）：
      對每一份文件的每個 chunk，呼叫 LLM 抽取 (subject, relation, object)
      三元組，存進 LlamaIndex 的 SimpleGraphStore，組成 G=(V,E)。

  - 查詢（query）：
      用 LlamaIndex 的 KGTableRetriever（retriever_mode="keyword"），
      先從 query 抽取關鍵詞（對應論文的 e_Ti），再沿著圖走訪深度 d
      （graph_store_query_depth=d），對應 Eq.(13) GraphTraversal(G, e_Ti, d)。

注意：三元組抽取 + 查詢時的關鍵詞抽取都會呼叫 LLM（透過
mmm_rag/llama_index_llm.py 代理到 model_manager 目前常駐的 Qwen2.5-3B），
所以建庫這一步（04_build_graph_index.py）會比舊版輕量版慢不少，
資料筆數多時請預留足夠時間。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.graph_stores import SimpleGraphStore

import config
from mmm_rag.llama_index_llm import ModelManagerLLM

# ------------------------------------------------------------------
# 全域 LlamaIndex 設定：LLM 走我們自己的 ModelManagerLLM（複用已載入的
# Qwen2.5-3B，不額外佔顯存）；embedding 這裡用不到（我們沒有開
# include_embeddings / retriever_mode="embedding"），明確關掉避免
# LlamaIndex 預設嘗試載入 OpenAI embedding 導致沒有 API key 而報錯。
# ------------------------------------------------------------------
Settings.llm = ModelManagerLLM()
Settings.embed_model = None


class GraphStore:
    def __init__(self):
        self.index: Optional[KnowledgeGraphIndex] = None

    def build_from_docs(self, docs: List[dict]):
        """
        docs: [{"doc_id": ..., "text": ...}, ...]
        對每份文件呼叫 LLM 抽取三元組，建出 KnowledgeGraphIndex（對應 G=(V,E)）。
        """
        documents = [
            Document(
                text=d["text"],
                id_=str(d["doc_id"]),
                metadata={"doc_id": str(d["doc_id"])},
            )
            for d in docs
            if d.get("text")
        ]
        if not documents:
            raise ValueError("沒有任何非空文件可以用來建立知識圖譜。")

        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=config.GRAPH_MAX_TRIPLETS_PER_CHUNK,
            include_embeddings=False,
            show_progress=True,
        )

    def describe(self) -> str:
        """
        回傳一段簡短的狀態摘要，給 main.py 開頭的「已載入資料庫」訊息用。

        舊版（networkx）GraphStore 有 .graph.number_of_nodes() 可以查，
        但這裡底層是 LlamaIndex 的 KnowledgeGraphIndex，內部結構在不同版本
        間可能有差異，所以刻意寫得保守：抓得到 docstore 就回報片段數，
        抓不到也不讓整個程式因為這種顯示用的小資訊而崩潰。
        """
        if self.index is None:
            return "尚未建立"
        try:
            n_docs = len(self.index.docstore.docs)
            return f"{n_docs} 個已索引文字片段"
        except Exception:  # noqa: BLE001
            return "已載入"

    def query(self, text: str, depth: int = 2, max_docs: int = 5) -> List[str]:
        """
        對應論文 Eq.(13): R_graph = GraphTraversal(G, e_Ti, d)。
        回傳走訪到、且能對應回原始文件的 doc_id 清單。
        """
        if self.index is None:
            return []

        retriever = self.index.as_retriever(
            retriever_mode="keyword",
            graph_store_query_depth=depth,
            similarity_top_k=max_docs,
            include_text=True,
        )
        nodes = retriever.retrieve(text)

        doc_ids: List[str] = []
        for n in nodes:
            doc_id = n.node.metadata.get("doc_id")
            if doc_id is not None and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        return doc_ids[:max_docs]

    def save(self, path: Path):
        """
        path 是一個資料夾路徑（取代舊版單一 .gpickle 檔）。
        LlamaIndex 會把 docstore / graph_store / index_store 分別存成
        好幾個 json 檔在這個資料夾底下。
        """
        if self.index is None:
            raise RuntimeError("尚未建立 index，無法儲存，請先呼叫 build_from_docs()。")
        path.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(path))

    @classmethod
    def load(cls, path: Path) -> "GraphStore":
        storage_context = StorageContext.from_defaults(persist_dir=str(path))
        store = cls()
        store.index = load_index_from_storage(storage_context)
        return store
