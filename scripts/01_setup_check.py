"""
01_setup_check.py
==================
在做任何下載/建庫動作之前，先跑這支腳本確認：
  1. venv 是否正確啟用
  2. PyTorch 是否抓得到 GPU（RTX 5060）與 CUDA
  3. 顯存大小是否符合預期（約 8GB）
  4. 這個專案要用到的套件是否都裝好了

用法（在專案根目錄、venv 啟用狀態下）：
    python scripts/01_setup_check.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers",
    "faiss",
    "datasets",
    "PIL",
    "ddgs",
    "qwen_vl_utils",
    "llama_index",
]


def check_python_version():
    print(f"[1] Python 版本: {sys.version}")
    if sys.version_info < (3, 10):
        print("    警告：建議使用 Python 3.10 或以上版本。")


def check_modules():
    print("\n[2] 套件檢查:")
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            print(f"    ✓ {mod}")
        except ImportError as e:
            print(f"    ✗ {mod}  --  {e}")
            missing.append(mod)
    if missing:
        print(f"\n    缺少套件: {missing}")
        print("    請先執行: pip install -r requirements.txt")
    return not missing


def check_cuda():
    print("\n[3] CUDA / GPU 檢查:")
    import torch

    print(f"    torch.__version__      = {torch.__version__}")
    print(f"    torch.cuda.is_available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = 0
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"    偵測到的顯卡           = {name}")
        print(f"    顯存總量                = {total_mem_gb:.1f} GB")
        if total_mem_gb < 7.0:
            print("    警告：顯存小於 7GB，建議在 config.py 把 DATASET_SUBSET_SIZE 調小，")
            print("          並確認 USE_4BIT = True、LOW_VRAM_MODE = True。")
    else:
        print("    沒有偵測到可用的 CUDA 顯卡。")
        print("    - 若你確定有安裝 NVIDIA 顯卡與驅動，請依 README 步驟 2 重新安裝對應 CUDA 版本的 torch。")
        print("    - 若只是想先測試程式邏輯，可以把 config.py 的 DEVICE 改成 'cpu'（會很慢）。")


def check_bnb():
    print("\n[4] bitsandbytes 4-bit 量化功能檢查:")
    try:
        import torch
        from transformers import BitsAndBytesConfig

        _ = BitsAndBytesConfig(load_in_4bit=True)
        if torch.cuda.is_available():
            print("    ✓ BitsAndBytesConfig 建立成功，且有 GPU 可用，4-bit 量化應可正常運作。")
        else:
            print("    ⚠ BitsAndBytesConfig 建立成功，但沒有 GPU，4-bit 量化在 CPU 上無意義。")
    except Exception as e:  # noqa: BLE001
        print(f"    ✗ 建立失敗: {e}")


if __name__ == "__main__":
    check_python_version()
    ok = check_modules()
    if ok:
        check_cuda()
        check_bnb()
    print("\n完成環境檢查。若上面全部是 ✓ / 顯示 CUDA 可用，就可以進入下一步 02_download_dataset.py。")
