"""
config.py
=========
所有可調參數集中在這裡。第一次跑之前建議先看過一遍，
尤其是「VRAM / 顯卡相關設定」區塊 —— 這是讓整個系統
塞進 RTX 5060 8GB 的關鍵開關。
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# 路徑設定
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.jsonl"      # 從 FinMME 整理出的 (I_db, T_db, ...) 清單

DB_DIR = ROOT_DIR / "db"
SEMANTIC_INDEX_PATH = DB_DIR / "semantic_feature_db.faiss"
SEMANTIC_META_PATH = DB_DIR / "semantic_feature_db.meta.pkl"
VISUAL_INDEX_PATH = DB_DIR / "visual_feature_db.faiss"
VISUAL_META_PATH = DB_DIR / "visual_feature_db.meta.pkl"
GRAPH_PATH = DB_DIR / "knowledge_graph_index"    # 資料夾：LlamaIndex storage_context.persist() 用

LOG_DIR = ROOT_DIR / "logs"

for p in (DATA_DIR, IMAGE_DIR, DB_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 資料集設定（luojunyu/FinMME）
# ------------------------------------------------------------------
HF_DATASET_NAME = "luojunyu/FinMME"
HF_DATASET_SPLIT = "train"

# FinMME 全量約 1.1 萬筆。8GB 顯卡上「建置資料庫」這一步需要跑 VLM 生成
# 每張圖的敘述，全量會非常久，所以預設只取前 N 筆做 demo。
# 想跑全量就設成 None。
DATASET_SUBSET_SIZE = 300

# ------------------------------------------------------------------
# 模型設定（全部是可以在 8GB VRAM 上以 4-bit 量化跑起來的小模型）
# 若你的顯卡 VRAM 更充裕，可以把這些換成同系列更大的模型
# （例如 Qwen2.5-7B-Instruct、Qwen2.5-VL-7B-Instruct），
# 架構完全不用改，只要換模型名稱即可。
# ------------------------------------------------------------------

# 文字型 LLM：負責 Task Decomposition Agent、
# Semantic/Web 子答案生成、Consensus Voting、最終答案整合
TEXT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 視覺語言模型 VLM：負責建庫時生成圖片語義描述 D(I)，
# 以及查詢時的 Visual 子答案 A_visual = VLM(I_j, T_j)
VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# CLIP：負責 Visual Feature Database 的雙塔編碼 (v_I, v_T)
CLIP_MODEL = "openai/clip-vit-base-patch32"

# 文字 Embedding 模型：負責 Semantic Feature Database 的稠密檢索
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------
# VRAM / 顯卡相關設定 —— RTX 5060 8GB 專用
# ------------------------------------------------------------------
USE_4BIT = True          # 開啟後 LLM / VLM 都以 4-bit (bitsandbytes NF4) 載入
DEVICE = "cuda"          # 若沒有 GPU 想先用 CPU 跑通流程，改成 "cpu"（會很慢，但能測試邏輯）

# 低 VRAM 模式：True 代表「同一時間只讓一個大模型（LLM 或 VLM）留在顯存」，
# 用到另一個時才切換載入 / 釋放。適合顯存吃緊、或你把模型換成更大版本時使用。
# False 代表 LLM + VLM + CLIP + Embedding 全部常駐（預設组合在 4-bit 下
# 實測約占 5~6GB，8GB 卡通常吃得下，速度也比一直切換模型快）。
LOW_VRAM_MODE = False

# ------------------------------------------------------------------
# 檢索超參數（對齊論文 Section II 的符號）
# ------------------------------------------------------------------
TOP_K = 3                 # 論文中的 top-K 檢索深度
GRAPH_TRAVERSAL_DEPTH = 2  # 論文 Eq.(13) 的 depth d
GRAPH_MAX_TRIPLETS_PER_CHUNK = 10  # LlamaIndex KnowledgeGraphIndex 每個 chunk 最多抽取的三元組數量
MAX_SUBTASKS = 3           # 論文 Eq.(7) 的 n <= 3
WEB_SEARCH_MAX_RESULTS = 5

# 生成長度上限
MAX_NEW_TOKENS_SHORT = 256   # 子答案 / 打分
MAX_NEW_TOKENS_LONG = 512    # 最終整合答案

# ------------------------------------------------------------------
# 其他
# ------------------------------------------------------------------
RANDOM_SEED = 42
HF_HOME = os.environ.get("HF_HOME", str(ROOT_DIR / ".hf_cache"))
os.environ.setdefault("HF_HOME", HF_HOME)
