# Changelog

## 2026-10-09 — 新增 FinMME 評估與消融實驗腳本

- **內容**：新增 `mmm_rag/evaluation.py`（題目格式化、選項／數值解析、判分、統計，純 Python）與 `scripts/05_evaluate.py`：取知識庫之後的 FinMME 題目（避免資料洩漏），以 `vlm_only`、`visual`、`semantic`、`web`、`full` 五種模式作答並計算各題型準確率，輸出 `results/eval_<mode>.jsonl` 與 `results/summary.md`。新增 `tests/test_evaluation.py`。README 加上評估說明。
- **原因**：repo 原本沒有任何量化結果。
- **測試**：`test_evaluation.py` 全部通過（含 "because" 中的 b 不被誤判為選項、多選需完全相同、數值 tolerance）；以假模型跑 `05_evaluate.py` 完整流程，正確寫出逐題紀錄與比較表。**實際評估需要 GPU，尚未執行**（此筆電為 MX150 2GB）。

## 2026-10-09 — repo 改名為 mmm-rag-finance

- **內容**：GitHub repo 由 `MMM-RAG-Stock-Prediction` 改名為 `mmm-rag-finance`（舊網址會自動轉址）。
- **原因**：本專案做的是財經圖表問答（FinMME），並沒有預測股價，舊名稱容易讓人誤會。
- **測試**：確認舊網址轉址到新網址。

## 2026-10-09 — 上傳完整實作程式碼

- **內容**：加入 MMM-RAG 完整實作（`mmm_rag/` 三個 Agent、向量庫、知識圖譜、模型管理；`scripts/01–04`；`main.py`；`config.py`）、`requirements.txt`、`.env.example`、中英雙語 README（含 Mermaid 架構圖、論文對照表）、`docs/setup-guide-zh.md`（原本的詳細安裝說明）、`tests/smoke_test.py`。
- **原因**：原 repo 只有空白 README，程式碼一直只在本機。
- **測試**：所有 `.py` 通過 `py_compile`；`tests/smoke_test.py` 以假模型通過（共識投票分數限制與權重正規化、子任務上限、JSON 解析保底）。**未測**：實際載入模型、下載 FinMME、建庫與端到端查詢（此電腦沒有 PyTorch／GPU 環境）。
