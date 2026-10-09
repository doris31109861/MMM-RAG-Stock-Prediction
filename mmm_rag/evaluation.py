"""
evaluation.py
=============
FinMME 評估用的純邏輯（不載入任何模型，可在沒有 GPU 的環境單元測試）。

FinMME 有三種題型：
  - single_choice   ：答案是一個選項字母，例如 "A"
  - multiple_choice ：答案是多個選項字母，例如 "ABCD"（順序不重要，要完全相同才算對）
  - numerical       ：答案是數字，tolerance 欄位為允許誤差（絕對值）；沒有 tolerance 時用 1% 相對誤差
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

CHOICE_TYPES = ("single_choice", "multiple_choice")


def format_question(record: Dict) -> str:
    """把資料集的一筆紀錄組成要問系統的問題，並要求固定的作答格式，方便自動判分。"""
    q = record.get("question_text", "").strip()
    qtype = record.get("question_type", "")
    if qtype in CHOICE_TYPES:
        how = "exactly one letter" if qtype == "single_choice" else "all correct letters (e.g. AC)"
        return (f"{q}\nOptions:\n{record.get('options', '').strip()}\n"
                f"Reply with {how} only, then a one-sentence reason.")
    unit = record.get("unit") or ""
    unit_hint = f" (unit: {unit})" if unit else ""
    return f"{q}{unit_hint}\nReply with a single number only, then a one-sentence reason."


def parse_choice(text: str, multiple: bool) -> str:
    """從模型輸出中取出選項字母（大寫、排序、去重）。

    先找 'Answer: X' 這類明確寫法，再找開頭的字母，最後退而求其次找第一個獨立的選項字母。
    """
    if not text:
        return ""
    t = text.strip()
    # 選項字母只認大寫，避免把 "because" 之類單字裡的 b 當成答案；只有 answer / is 不分大小寫
    letters_re = r"([A-F](?:(?:[\s,、]|and|和)*[A-F])*)"   # 允許 "A, B and D"、"A 和 C"
    end = r"(?=[\s.:)、,，。]|$)"
    m = re.search(r"(?i:answer)\s*(?i:is)?\s*[:：]?\s*\(?" + letters_re + r"\)?" + end, t)
    if not m:
        m = re.match(r"\(?" + letters_re + r"\)?" + end, t)
    if not m:
        m = re.search(r"\b([A-F])\b", t)
    if not m:
        return ""
    letters = sorted(set(re.findall(r"[A-F]", m.group(1).upper())))
    if not multiple:
        letters = letters[:1]
    return "".join(letters)


def parse_number(text: str) -> Optional[float]:
    """取出模型輸出中的第一個數字（支援負號、千分位逗號、百分比符號）。"""
    if not text:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*|-?\.\d+", text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def is_correct(record: Dict, prediction_text: str) -> bool:
    """判斷模型的作答是否正確。"""
    qtype = record.get("question_type", "")
    answer = str(record.get("answer", "")).strip()
    if qtype in CHOICE_TYPES:
        pred = parse_choice(prediction_text, multiple=(qtype == "multiple_choice"))
        return pred != "" and pred == "".join(sorted(set(answer.upper())))
    pred = parse_number(prediction_text)
    try:
        gold = float(answer)
    except ValueError:
        return False
    if pred is None:
        return False
    tol = record.get("tolerance")
    if tol is None:
        tol = abs(gold) * 0.01           # 沒有給允許誤差時，用 1% 相對誤差
    return abs(pred - gold) <= float(tol) + 1e-9


def summarize(results: List[Dict]) -> Dict:
    """統計整體與各題型的準確率。results 每筆需有 question_type 與 correct。"""
    def acc(rows):
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None

    by_type = {}
    for qtype in sorted({r["question_type"] for r in results}):
        rows = [r for r in results if r["question_type"] == qtype]
        by_type[qtype] = {"n": len(rows), "accuracy": acc(rows)}
    return {"n": len(results), "accuracy": acc(results), "by_type": by_type}
