"""
tests/test_evaluation.py — 評估判分邏輯的單元測試（只需 Python 標準函式庫）

用法：python tests/test_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmm_rag.evaluation import format_question, is_correct, parse_choice, parse_number, summarize  # noqa: E402

SINGLE = {"question_type": "single_choice", "answer": "B", "question_text": "Q?", "options": "A: x\nB: y"}
MULTI = {"question_type": "multiple_choice", "answer": "ABD"}
NUM_TOL = {"question_type": "numerical", "answer": "12.5", "tolerance": 5}
NUM_NOTOL = {"question_type": "numerical", "answer": "200", "tolerance": None}


def test_parse_choice():
    assert parse_choice("B", False) == "B"
    assert parse_choice("B. Because the red line rises", False) == "B"
    assert parse_choice("A Because ...", False) == "A"           # Because 的 B 不能被當成答案
    assert parse_choice("The answer is C.", False) == "C"
    assert parse_choice("Answer: (D)", False) == "D"
    assert parse_choice("because it goes up", False) == ""       # 沒有大寫選項字母
    assert parse_choice("A, B and D", True) == "ABD"             # 支援 and 連接
    assert parse_choice("A and Because", False) == "A"
    assert parse_choice("ABD", True) == "ABD"
    assert parse_choice("D, A, B", True) == "ABD"                # 排序後比較
    assert parse_choice("", True) == ""


def test_parse_number():
    assert parse_number("12.3 because") == 12.3
    assert parse_number("about -4%") == -4.0
    assert parse_number("1,234.5 million") == 1234.5
    assert parse_number("no number") is None


def test_is_correct():
    assert is_correct(SINGLE, "B. the second one")
    assert not is_correct(SINGLE, "A")
    assert is_correct(MULTI, "Answer: ABD")
    assert not is_correct(MULTI, "AB")                           # 少選一個就算錯
    assert is_correct(NUM_TOL, "16 percent")                      # |16 - 12.5| <= 5
    assert not is_correct(NUM_TOL, "18")
    assert is_correct(NUM_NOTOL, "201.5")                         # 1% 相對誤差內
    assert not is_correct(NUM_NOTOL, "203")


def test_format_question():
    q = format_question(SINGLE)
    assert "Options:" in q and "exactly one letter" in q
    assert "single number" in format_question({"question_type": "numerical", "question_text": "How much?", "unit": "%"})


def test_summarize():
    s = summarize([{"question_type": "single_choice", "correct": True},
                   {"question_type": "single_choice", "correct": False},
                   {"question_type": "numerical", "correct": True}])
    assert s["n"] == 3 and abs(s["accuracy"] - 0.6667) < 1e-4
    assert s["by_type"]["single_choice"] == {"n": 2, "accuracy": 0.5}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
