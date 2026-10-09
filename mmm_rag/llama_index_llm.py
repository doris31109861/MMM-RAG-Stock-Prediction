"""
llama_index_llm.py
====================
把 model_manager 目前管理的文字 LLM（Qwen2.5-3B-Instruct）包裝成
LlamaIndex 認得的 LLM 介面（CustomLLM），這樣 graph_store.py 用
LlamaIndex 的 KnowledgeGraphIndex 做「三元組抽取」以及查詢時的
「關鍵詞抽取」，都會直接呼叫同一個已經常駐 VRAM 的模型，
不會因為引入 LlamaIndex 而額外多載一份 LLM 佔用顯存。
"""

from __future__ import annotations

from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

import config


class ModelManagerLLM(CustomLLM):
    """代理到 mmm_rag.model_manager.model_manager 的 LlamaIndex LLM。"""

    context_window: int = 4096
    num_output: int = 256
    model_name: str = config.TEXT_LLM_MODEL

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 延遲 import，避免 model_manager / graph_store 互相 import 造成循環引用
        from mmm_rag.model_manager import model_manager

        text = model_manager.generate_text(prompt, max_new_tokens=self.num_output)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # 本地模型沒有做真正的串流生成，這裡用一次性結果模擬「單次 yield」的串流介面，
        # 滿足 LlamaIndex 的 API 需求即可（三元組抽取/關鍵詞抽取都是非串流用途）。
        response = self.complete(prompt, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()
