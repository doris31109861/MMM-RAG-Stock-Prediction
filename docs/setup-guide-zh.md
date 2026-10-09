# MMM-RAG 本地實作（財經 / 股市版，RTX 5060 8GB）

這個專案把 ICPADS 2025 論文
**《MMM-RAG: A Multi-Agent Multi-Feature Method for Multimodal Retrieval-Augmented Generation》**
的架構，改用能塞進單張 **RTX 5060 8GB** 消費級顯卡的小模型重新實作一遍，
知識庫資料來源是 [`luojunyu/FinMME`](https://huggingface.co/datasets/luojunyu/FinMME)
（一個財經圖表多模態問答資料集，11,000+ 筆研究報告圖表 + 問題）。

> ⚠️ **這是一個「架構教學 / demo」專案，不是投資建議工具**。目標是忠實重現論文的
> 系統設計（雙特徵資料庫、三個 Agent、共識投票），**不追求答案的正確率**。
> 所有的財經/股市內容都只是拿來當作多模態 RAG 的示範素材。

---

## 0. 架構對照表

在動手之前，先建立「論文寫了什麼」跟「這個專案哪個檔案對應」的心智地圖：

| 論文元件 (Fig.1 / Fig.2) | 論文公式 | 本專案對應檔案 | 本地替代模型 |
|---|---|---|---|
| Feature-Parsing Agent（建 Semantic/Visual DB） | Eq.(1)(2)(3)(4)(5) | `scripts/03_build_knowledge_base.py` | Qwen2-VL-2B（描述）+ CLIP ViT-B/32（雙塔編碼）+ BGE-small（文字向量） |
| Semantic Feature Database | Eq.(2) | `mmm_rag/vector_store.py` + `db/semantic_feature_db.faiss` | FAISS |
| Visual Feature Database | Eq.(5) | `mmm_rag/vector_store.py` + `db/visual_feature_db.faiss` | FAISS |
| Task Decomposition Agent | Eq.(6)(7) | `mmm_rag/decomposition_agent.py` | Qwen2.5-3B-Instruct |
| Visual retrieval agent | Eq.(8)(9)(10) | `retrieval_agent.retrieve_visual` | CLIP + Qwen2-VL-2B |
| Semantic retrieval agent（dense + graph） | Eq.(11)(12)(13)(14) | `retrieval_agent.retrieve_semantic` | BGE + `mmm_rag/graph_store.py`（LlamaIndex `KnowledgeGraphIndex`）+ Qwen2.5-3B |
| Web retrieval agent | Eq.(15) | `retrieval_agent.retrieve_web` | `ddgs`（DuckDuckGo）+ Qwen2.5-3B |
| Consensus Voting and Answer Generation | Eq.(16)(17)(18) | `mmm_rag/decision_agent.py` | Qwen2.5-3B-Instruct |

原論文用 Qwen2.5-VL-7B + Qwen3-14B + BLIP-2 + LlamaIndex 知識圖譜，
文字/視覺/影像編碼這幾個模型在 8GB 顯卡上塞不下，所以換成同系列的小模型，
**函式的輸入輸出關係跟論文公式一一對應**。graph retrieval 這部分則保留了論文
原本的做法：用 LlamaIndex 的 `KnowledgeGraphIndex` 真的呼叫 LLM 抽取三元組建圖，
查詢時也真的呼叫 LLM 抽關鍵詞做 `GraphTraversal(G, e_Ti, d)`（Eq.13）。
唯一的差異是這裡沒有讓 LlamaIndex 自己另外載一個模型，而是透過
`mmm_rag/llama_index_llm.py` 把它要用的 LLM 代理到 `model_manager` 已經
常駐顯存的 Qwen2.5-3B，避免多佔一份 VRAM（見第 10 節 FAQ Q5）。

---

## 1. 前置需求

- **作業系統**：Windows 10/11 或 Linux 皆可（以下指令兩種都會附）
- **PyCharm**：Community 或 Professional 版都可以
- **Python 3.10 或 3.11**（建議別用 3.13，部分套件的預編譯 wheel 還沒跟上）
- **NVIDIA 顯卡驅動**：請先到 NVIDIA 官網更新到最新版驅動（RTX 5060 是 Blackwell 架構，
  舊驅動可能連顯卡都認不出來）
- 至少 **20GB 左右的硬碟空間**（模型快取 + 資料集圖片）
- 穩定的網路（第一次會下載約 5-8GB 的模型 + 資料集）

---

## 2. 步驟一：在 PyCharm 建立虛擬環境（venv）

這一步的目的是讓這個專案的套件全部裝在一個獨立資料夾裡，
**不會汙染你電腦上其他 Python 專案或系統 Python**。

1. 打開 PyCharm → `File > Open...`，選擇你要放這個專案的資料夾（例如把這整包檔案解壓縮後的
   `mmm_rag_stock` 資料夾）。
2. 打開後，PyCharm 通常會自動偵測沒有直譯器並跳出提示；如果沒有，手動設定：
   `File > Settings > Project: mmm_rag_stock > Python Interpreter`
3. 點右上角齒輪 `Add Interpreter > Add Local Interpreter`
4. 選擇 **Virtualenv Environment > New**：
   - Location：預設會是 `專案路徑\.venv`，保持預設即可
   - Base interpreter：選擇你電腦上已安裝的 Python 3.10 / 3.11
   - 按 OK，PyCharm 會在專案資料夾下建立一個 `.venv` 資料夾
5. 建立完成後，打開 PyCharm 下方的 **Terminal** 分頁，確認提示字元前面出現
   `(.venv)` 字樣，代表虛擬環境已啟用。之後所有指令都在這個 Terminal 裡執行。

> 如果你偏好純命令列（不透過 PyCharm UI），效果完全一樣：
> ```bash
> # Windows (PowerShell)
> python -m venv .venv
> .venv\Scripts\activate
>
> # Linux / macOS
> python3 -m venv .venv
> source .venv/bin/activate
> ```
> 建立好之後回到 PyCharm，用上面步驟 2-3 把 `Existing environment` 指向這個 `.venv` 即可。

---

## 3. 步驟二：安裝 PyTorch（RTX 5060 專用，這步最容易踩雷）

**請不要直接 `pip install torch`**，也不要放進 `requirements.txt` 一起裝。

RTX 5060 屬於 NVIDIA **Blackwell 架構**，CUDA compute capability 是 `sm_120`。
如果裝到沒有支援 `sm_120` 內核的 PyTorch 版本，`torch.cuda.is_available()`
會顯示 `True`（因為驅動看得到顯卡），但實際做運算時會報：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

解法是安裝 **PyTorch 2.8（或更新）+ CUDA 12.8 (cu128)** 對應的版本。
在 venv 啟用的狀態下，於 Terminal 執行：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

裝完後立刻驗證：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

正常應該會印出類似：

```
2.8.0+cu128 True NVIDIA GeForce RTX 5060
```

> **如果還是出現 `sm_120 not compatible`**：代表你裝到的還是舊版 wheel。可以改試官方
> nightly 版：
> ```bash
> pip uninstall torch torchvision torchaudio -y
> pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
> ```
> 這個問題屬於 PyTorch 官方持續在更新的範圍，若上面兩個指令都不行，
> 到 https://pytorch.org/get-started/locally/ 選擇你的作業系統 / CUDA 版本，
> 網站會給你當下最新、正確的安裝指令。

---

## 4. 步驟三：安裝其餘套件

確認 torch 能抓到 GPU 之後，安裝其他所有套件：

```bash
pip install -r requirements.txt
```

這會裝：`transformers`、`accelerate`、`bitsandbytes`（4-bit 量化用）、
`sentence-transformers`、`faiss-cpu`、`llama-index`（知識圖譜，`KnowledgeGraphIndex` +
`SimpleGraphStore`）、`datasets` / `huggingface_hub`（下載 FinMME）、`ddgs`（網路搜尋，
原本叫 `duckduckgo_search`，2025 年改名為 `ddgs`）等等。

> 如果你的 `requirements.txt` 是比較舊的版本、還沒有加 `llama-index`，
> 先手動在裡面補一行 `llama-index`，再重新執行 `pip install -r requirements.txt`；
> 或者直接 `pip install llama-index` 補裝也可以。

> **Windows 上 `bitsandbytes` 安裝失敗怎麼辦？**
> 新版 `bitsandbytes`（>=0.43）已經有官方 Windows wheel，一般 `pip install` 就能裝。
> 如果還是失敗，先確認 pip 版本夠新：`python -m pip install --upgrade pip`，
> 再重新執行 `pip install -r requirements.txt`。

---

## 5. 步驟四：環境檢查

在真的下載模型跟資料集之前，先跑這支腳本，一次確認 Python / CUDA / 套件都沒問題：

```bash
python scripts/01_setup_check.py
```

理想輸出：套件檢查全部 `✓`、`torch.cuda.is_available = True`、
顯存大約 `8.0 GB`、bitsandbytes 4-bit 檢查成功。
**任何一項失敗都先解決再往下走**，不然後面下載模型會白等。

---

## 6. 步驟五：下載 FinMME 資料集

```bash
python scripts/02_download_dataset.py
```

這支腳本會：

1. 用 `datasets.load_dataset("luojunyu/FinMME", split="train")` 從 HuggingFace 下載資料集
   （第一次執行會把資料快取到 `.hf_cache/`，之後重跑不會重複下載）
2. 把每一筆的圖片存成 `data/images/{id}.jpg`
3. 把 `verified_caption`、`related_sentences`（對應論文的 T_db，也就是「與影像關聯的原始文本說明」）、
   以及 `question_text` / `options` / `answer` 等欄位整理成 `data/metadata.jsonl`

預設只抓 **前 300 筆**（`config.py` 的 `DATASET_SUBSET_SIZE`），因為下一步要對每張圖跑一次
VLM 生成描述，全量 1.1 萬筆在單張消費級顯卡上會跑非常久。想跑全量／改成別的數量，
直接改 `config.py`：

```python
DATASET_SUBSET_SIZE = 300   # 改成你想要的數字，或改成 None 代表跑全量
```

---

## 7. 步驟六：建置雙特徵知識庫（Feature-Parsing Agent）

這一步對應論文 **Section II-A**，也是整個系統裡最花時間的一步（要對每張圖跑一次 VLM）：

```bash
python scripts/03_build_knowledge_base.py
```

它會依序做：

- **Semantic Feature Database**：用 Qwen2-VL-2B 幫每張圖生成一段描述 `D(I)`（對應 Eq.1），
  跟資料集原本的文字說明 `T` 串接成 `S = Concat(T, D(I))`（Eq.2），
  再用 `BAAI/bge-small-en-v1.5` 把 `S` 轉成向量，存進 `db/semantic_feature_db.faiss`
- **Visual Feature Database**：用 CLIP 的影像塔編碼圖片 `v_I`（Eq.3）、文字塔編碼 `T`
  得到 `v_T`（Eq.4），兩者串接成 `F`（Eq.5），存進 `db/visual_feature_db.faiss`

300 筆資料在 RTX 5060 上大約需要 **10-25 分鐘**（實際時間依模型下載速度、
圖片解析度而定；第一次執行還要另外下載 Qwen2-VL-2B / CLIP / BGE 模型權重，約 5-6GB）。

接著建立知識圖譜（對應論文的 graph-based retrieval，Eq.13）：

```bash
python scripts/04_build_graph_index.py
```

這一步會用 LlamaIndex 的 `KnowledgeGraphIndex` 對每個文字 chunk 呼叫一次 LLM
（Qwen2.5-3B）抽取三元組建圖，**不是純 CPU 規則處理**，所以不會像舊版那樣
幾秒鐘完成——300 筆資料大約需要幾分鐘（比 03 快，但明顯比舊版慢）。
會產生一個資料夾 `db/knowledge_graph_index/`（裡面是 LlamaIndex 存的
`graph_store.json` / `docstore.json` / `index_store.json` 等檔案），
取代舊版單一的 `knowledge_graph.gpickle`。

> 如果你之前用舊版程式跑過 `04_build_graph_index.py`，資料夾裡可能還留著
> 舊的 `knowledge_graph.gpickle`，可以直接刪掉（新版不會讀它），
> 重新執行這支腳本產生新格式的 `knowledge_graph_index/`。

跑完這兩支之後，`db/` 資料夾應該長這樣：

```
db/
├── semantic_feature_db.faiss
├── semantic_feature_db.meta.pkl
├── visual_feature_db.faiss
├── visual_feature_db.meta.pkl
└── knowledge_graph_index/       <- LlamaIndex storage_context.persist() 產生的資料夾
    ├── graph_store.json
    ├── docstore.json
    └── ...（其他 LlamaIndex 內部索引檔）
```

---

## 8. 步驟七：執行查詢（三個 Agent 的完整流程）

一切就緒後，跑一次端到端查詢：

```bash
python main.py --question "從這批財經圖表資料庫來看，分析師在解讀個股股價走勢圖時，通常會關注哪些跨期趨勢與相對大盤表現的訊號？"
```

或者偷懶一點，直接從 FinMME 隨機抽一筆現成問題來測試整條 pipeline 能不能跑通：

```bash
python main.py --sample
```

**你也可以附上自己的一張圖**（例如自己截的股價走勢圖）：

```bash
python main.py --image "C:\path\to\your_chart.png" --question "這張圖的走勢有什麼值得注意的地方？"
```

有 `--image` 的時候：

- Task Decomposition Agent 會先用 VLM 把這張圖轉成文字描述，讓拆子任務時也考慮圖片內容
  （見 `decomposition_agent.py` 的 `image_caption`）
- Visual retrieval 會直接用這張圖真正的 CLIP 影像向量去比對資料庫（而不是像純文字問題那樣
  用文字向量頂替），找出最相似的參考圖表後，把「你的圖」跟「資料庫裡最相似的參考圖」
  兩張一起丟給 VLM 做比較
- 沒有 `--image` 時，一切照舊只用文字運作，不受影響

執行過程你會依序看到：

1. **`[Step 1]`** Task Decomposition Agent 判斷是否為複雜查詢、要不要拆成子任務（Eq.6/7）
2. **`[Step 2+3]`** 針對每個子任務，同時跑 visual / semantic / web 三路檢索拿到三個候選答案，
   再由 Consensus Voting 打分（Eq.16）、算出權重（Eq.17）、融合出這個子任務的答案
3. 若有多個子任務，最後再用 LLM 把各子任務答案整合成對原始問題的最終回答（Eq.18 的延伸）

完整的中間過程（每個候選答案、分數、權重）都會存成 JSON，路徑印在最後一行，
存放於 `logs/run_<timestamp>.json`，方便你檢查每個 Agent 到底做了什麼決策。

---

## 9. 專案結構總覽

```
mmm_rag_stock/
├── README.md                    <- 這份文件
├── requirements.txt
├── config.py                    <- 所有可調參數（模型名稱、VRAM 模式、TOP_K...）
├── main.py                      <- 端到端查詢入口
├── data/
│   ├── images/                  <- FinMME 圖片 (I_db)
│   └── metadata.jsonl           <- FinMME 文字說明 + 問題 (T_db, Q)
├── db/
│   ├── semantic_feature_db.*    <- 語義特徵資料庫 (FAISS)
│   ├── visual_feature_db.*      <- 視覺特徵資料庫 (FAISS)
│   └── knowledge_graph_index/   <- 知識圖譜 (LlamaIndex KnowledgeGraphIndex 持久化資料夾)
├── logs/                        <- 每次查詢的完整過程紀錄 (JSON)
├── scripts/
│   ├── 01_setup_check.py        <- 環境檢查
│   ├── 02_download_dataset.py   <- 下載 FinMME
│   ├── 03_build_knowledge_base.py  <- 建置雙特徵資料庫
│   └── 04_build_graph_index.py  <- 建置知識圖譜
└── mmm_rag/
    ├── model_manager.py         <- 統一管理 LLM / VLM / CLIP / Embedding 模型（含 4-bit 量化）
    ├── vector_store.py          <- FAISS 向量庫包裝
    ├── graph_store.py           <- 知識圖譜（LlamaIndex KnowledgeGraphIndex，真的呼叫 LLM 抽三元組）
    ├── llama_index_llm.py       <- 把 LlamaIndex 要用的 LLM 代理到 model_manager，避免重複佔 VRAM
    ├── decomposition_agent.py   <- Task Decomposition Agent
    ├── retrieval_agent.py       <- Multi-Feature Multi-Source Retrieval Agent
    └── decision_agent.py        <- Consensus Voting and Answer Generation Agent
```

---

## 10. VRAM 調校 / 常見問題 FAQ

**Q1：跑 `03_build_knowledge_base.py` 或 `main.py` 時 CUDA Out of Memory？**

打開 `config.py`，做以下調整：

```python
LOW_VRAM_MODE = True       # 改成 True：LLM / VLM 用完就卸載，不會同時常駐顯存
DATASET_SUBSET_SIZE = 100  # 建庫的資料量再調小一點
```

`LOW_VRAM_MODE = True` 時，`ModelManager` 每次呼叫完文字 LLM 或 VLM 後會自動
`unload` 並清空 CUDA 快取，下次要用再重新載入 —— 速度會變慢（重複載入模型），
但顯存壓力最小，這是 8GB 卡最保險的組合。

**Q2：想要答案品質好一點，該怎麼換更大的模型？**

`config.py` 裡的四個模型名稱都可以直接換成同系列更大的版本，不用改任何架構程式碼：

```python
TEXT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"        # 原本是 3B
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"          # 原本是 Qwen2-VL-2B（更貼近論文原設定）
```

換成 7B 級模型後，建議同時把 `LOW_VRAM_MODE` 設成 `True`（兩個 7B 模型 4-bit
同時常駐可能會超過 8GB），並預期建庫 / 查詢速度會明顯變慢。

**Q3：`ddgs`（網路檢索）常常抓不到結果或被限流？**

DuckDuckGo 沒有官方 API，`ddgs` 是社群套件，對同一組關鍵字太頻繁查詢時
可能會回空結果或暫時被擋。`retrieval_agent.retrieve_web()` 已經包了
`try/except`，抓不到時會回傳「(網路檢索失敗)」而不會讓整個程式當掉，
單純是那個子答案的品質會比較差 —— 這在示範專案裡是可以接受的。

**Q4：為什麼 Visual Feature Database 用 CLIP，不是論文的 BLIP-2？**

論文用 BLIP-2 的 ViT + 對應文字編碼器讓兩個塔天生對齊在同一個語義空間。
CLIP（`openai/clip-vit-base-patch32`）做的是完全一樣的事情：影像塔 + 文字塔
共享對齊過的向量空間，模型小很多（約 600MB vs BLIP-2 的數 GB），
很適合 8GB 顯卡，架構上的角色（Eq.3/4/5）完全對應，只是換一個更輕量的雙塔模型。

**Q5：知識圖譜檢索（graph-based retrieval）是不是真的有呼叫 LLM 抽三元組？**

是的。`mmm_rag/graph_store.py` 用 LlamaIndex 的 `KnowledgeGraphIndex`：
建庫時（`04_build_graph_index.py`）對每個文字 chunk 呼叫一次 LLM 抽取
`(subject, relation, object)` 三元組，組成 G=(V,E)；查詢時用
`KGTableRetriever`（`retriever_mode="keyword"`），先呼叫 LLM 從問題裡抽出
關鍵詞（對應論文的 entity e_Ti），再沿著圖做深度 `d`
（`config.GRAPH_TRAVERSAL_DEPTH`）的走訪，對應 Eq.(13)
`GraphTraversal(G, e_Ti, d)`——跟論文原本的做法一致。

這兩處 LLM 呼叫都不是額外載入新模型：透過 `mmm_rag/llama_index_llm.py`
裡的 `ModelManagerLLM`，LlamaIndex 會直接代理到 `model_manager` 已經
常駐顯存的 Qwen2.5-3B，不會因為引入 LlamaIndex 而多佔一份 VRAM。
代價是建庫（04）跟查詢（每個 subtask 的 semantic retrieval）都會比
純規則式關鍵詞抽取多幾次 LLM 呼叫，速度會慢一些——這是換成「忠於論文」
做法必然的取捨，如果你的顯卡/時間比較吃緊，也可以把
`config.GRAPH_MAX_TRIPLETS_PER_CHUNK` 調小、或 `GRAPH_TRAVERSAL_DEPTH`
調成 1，減少三元組數量與走訪範圍。

---

## 11. 延伸：換成別的知識庫主題

想把知識庫換成別的財經資料（例如你自己爬的個股新聞、財報 PDF 截圖等），
只要準備好跟 `data/metadata.jsonl` 一樣的格式（`doc_id` / `image_path` /
一段文字說明），從 `03_build_knowledge_base.py` 開始重跑即可，
`02_download_dataset.py` 只是把 FinMME 轉成這個格式的其中一種實作方式。

---

## 免責聲明

本專案僅為 MMM-RAG 論文架構的教學重現，使用的財經圖表資料與生成的回答
**不構成任何投資建議**，模型也未針對正確率做任何調校或微調。
