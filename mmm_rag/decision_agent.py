"""
decision_agent.py
==================
對應論文 Section II-D「Consensus Voting and Answer Generation」。

  1) Structured Consistency Judgement -> Eq.(16): LLM 對三個候選答案
     各自在 Consistency / Coherence / Relevance / Credibility 四個面向打 1-5 分，
     加總/平均成一個信心分數 c_visual, c_semantic, c_web。
  2) Weighted Consensus Voting -> Eq.(17): 依信心分數正規化成權重 w_j，
     選出（或用權重提示 LLM 融合出）最終答案。
  3) 最終再讓 LLM 看過 Q + 三個候選 + 初步共識答案，輸出精煉後的最終回答 -> Eq.(18)。
"""

from __future__ import annotations

import json
import re
from typing import Dict

import config
from mmm_rag.model_manager import model_manager

_SYSTEM_PROMPT = (
    "You are the Consensus Voting and Answer Generation Agent in a multimodal "
    "financial RAG system. Respond with STRICT JSON only, no markdown fences."
)

_SCORE_PROMPT = """Task: Evaluate the following question and candidate answers.

Question: {question}

Candidate-1 (visual): {a_visual}
Candidate-2 (semantic): {a_semantic}
Candidate-3 (web): {a_web}

Rate each candidate on a 1-5 scale for:
1. Consistency: alignment with the question intent
2. Coherence: clarity and logical self-consistency
3. Relevance: topical pertinence
4. Credibility: trustworthiness and evidential support

Return the AVERAGE of the four scores for each candidate (a single number per candidate,
between 1 and 5), as JSON exactly in this schema:
{{"visual": <number>, "semantic": <number>, "web": <number>}}
"""

_FINAL_PROMPT = """You are given a question, three candidate answers from different retrieval
sources, and their confidence weights (higher = more trustworthy). Synthesize ONE final,
concise, well-written answer. If candidates disagree, prefer the higher-weighted ones,
but reconcile them into a single coherent answer rather than just repeating the best one.

Question: {question}

Candidate-1 (visual, weight={w_visual:.2f}): {a_visual}
Candidate-2 (semantic, weight={w_semantic:.2f}): {a_semantic}
Candidate-3 (web, weight={w_web:.2f}): {a_web}

Final answer:"""


def _extract_json(raw: str) -> Dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


def _default_scores() -> Dict[str, float]:
    # 三者一視同仁的保底分數（LLM 打分失敗時使用）
    return {"visual": 3.0, "semantic": 3.0, "web": 3.0}


def score_candidates(question: str, candidates: Dict[str, str]) -> Dict[str, float]:
    """對應 Eq.(16)，回傳 {"visual": c, "semantic": c, "web": c}。"""
    prompt = _SCORE_PROMPT.format(
        question=question,
        a_visual=candidates.get("visual", ""),
        a_semantic=candidates.get("semantic", ""),
        a_web=candidates.get("web", ""),
    )
    raw = model_manager.generate_text(
        prompt, system=_SYSTEM_PROMPT, max_new_tokens=config.MAX_NEW_TOKENS_SHORT
    )
    try:
        parsed = _extract_json(raw)
        scores = {
            k: float(parsed.get(k, 3.0))
            for k in ("visual", "semantic", "web")
        }
        for k, v in scores.items():
            scores[k] = min(max(v, 1.0), 5.0)
        return scores
    except Exception as e:  # noqa: BLE001
        print(f"[DecisionAgent] 打分 JSON 解析失敗（{e}），使用預設分數。原始輸出：{raw!r}")
        return _default_scores()


def weighted_consensus(scores: Dict[str, float]) -> Dict[str, float]:
    """對應 Eq.(17) 的 w_j = c_j / sum(c_k)。"""
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def consensus_vote_and_generate(question: str, candidates: Dict[str, Dict]) -> Dict:
    """
    candidates: {"visual": {"answer": ..., "refs": [...]}, "semantic": {...}, "web": {...}}
    回傳最終整合結果，對應 Eq.(18) A_output。
    """
    plain_candidates = {k: v.get("answer", "") for k, v in candidates.items()}

    scores = score_candidates(question, plain_candidates)
    weights = weighted_consensus(scores)

    best_source = max(weights, key=weights.get)

    final_prompt = _FINAL_PROMPT.format(
        question=question,
        w_visual=weights.get("visual", 0.0),
        w_semantic=weights.get("semantic", 0.0),
        w_web=weights.get("web", 0.0),
        a_visual=plain_candidates.get("visual", ""),
        a_semantic=plain_candidates.get("semantic", ""),
        a_web=plain_candidates.get("web", ""),
    )
    final_answer = model_manager.generate_text(final_prompt, max_new_tokens=config.MAX_NEW_TOKENS_LONG)

    return {
        "final_answer": final_answer,
        "scores": scores,
        "weights": weights,
        "best_source": best_source,
        "candidates": candidates,
    }
