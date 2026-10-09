# Changelog

## 2026-10-09 — 上傳完整實作程式碼

- **內容**：加入 MMM-RAG 完整實作（`mmm_rag/` 三個 Agent、向量庫、知識圖譜、模型管理；`scripts/01–04`；`main.py`；`config.py`）、`requirements.txt`、`.env.example`、中英雙語 README（含 Mermaid 架構圖、論文對照表）、`docs/setup-guide-zh.md`（原本的詳細安裝說明）、`tests/smoke_test.py`。
- **原因**：原 repo 只有空白 README，程式碼一直只在本機。
- **測試**：所有 `.py` 通過 `py_compile`；`tests/smoke_test.py` 以假模型通過（共識投票分數限制與權重正規化、子任務上限、JSON 解析保底）。**未測**：實際載入模型、下載 FinMME、建庫與端到端查詢（此電腦沒有 PyTorch／GPU 環境）。
