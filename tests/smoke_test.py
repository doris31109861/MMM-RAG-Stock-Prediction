"""
tests/smoke_test.py
===================
不需要 GPU、不下載任何模型的冒煙測試（smoke test）。

用假的 model_manager 取代真正的 LLM / VLM，固定回傳預先寫好的字串，
藉此檢查 Agent 的「程式邏輯」是否正確：
  - Consensus Voting：分數會被限制在 1~5、權重 w_j = c_j / Σc_k 加總為 1（論文 Eq.16、17）
  - Task Decomposition：子任務數量上限 n <= MAX_SUBTASKS（論文 Eq.7）、JSON 解析失敗時的保底邏輯

用法（只需要 Python 標準函式庫）：
    python tests/smoke_test.py
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 以假模組取代 PIL 與 model_manager，避免載入 torch / transformers ----
_pil = types.ModuleType("PIL")
_pil.Image = types.SimpleNamespace(Image=object)
sys.modules.setdefault("PIL", _pil)


class _FakeModelManager:
    """依 prompt 內容回傳固定答案，模擬 LLM 的輸出（包含不合規範的輸出）。"""

    def __init__(self):
        self.decompose_reply = '{"is_complex": true, "subtasks": ["a", "b", "c", "d"]}'

    def generate_text(self, prompt, system=None, max_new_tokens=0):
        if "Rate each candidate" in prompt:
            # 前面夾雜多餘文字、web 分數超出範圍（9），測試 JSON 擷取與 clamp
            return 'Sure! {"visual": 4, "semantic": 2, "web": 9}'
        if "decide:" in prompt:
            return self.decompose_reply
        return "final answer"


_fake_mm = _FakeModelManager()
_mm_module = types.ModuleType("mmm_rag.model_manager")
_mm_module.model_manager = _fake_mm
sys.modules["mmm_rag.model_manager"] = _mm_module

import config  # noqa: E402
from mmm_rag import decision_agent, decomposition_agent  # noqa: E402


def test_consensus_voting():
    scores = decision_agent.score_candidates("q", {"visual": "x", "semantic": "y", "web": "z"})
    assert scores == {"visual": 4.0, "semantic": 2.0, "web": 5.0}, scores  # 9 被限制成 5

    weights = decision_agent.weighted_consensus(scores)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert max(weights, key=weights.get) == "web"

    result = decision_agent.consensus_vote_and_generate(
        "q", {k: {"answer": k} for k in ("visual", "semantic", "web")}
    )
    assert result["best_source"] == "web"
    assert result["final_answer"] == "final answer"


def test_decomposition_limits_subtasks():
    out = decomposition_agent.decompose_query("q")
    assert out["is_complex"] is True
    assert len(out["subtasks"]) == config.MAX_SUBTASKS  # 4 個被截成 3 個


def test_decomposition_fallback_on_bad_json():
    _fake_mm.decompose_reply = "I cannot answer in JSON"
    try:
        out = decomposition_agent.decompose_query("original question")
    finally:
        _fake_mm.decompose_reply = '{"is_complex": true, "subtasks": ["a", "b", "c", "d"]}'
    # 解析失敗時視為非複雜查詢，原問題本身就是唯一的子任務
    assert out == {"is_complex": False, "subtasks": ["original question"], "image_caption": ""}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
