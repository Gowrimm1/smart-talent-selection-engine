"""
ranker.py — Calibrated Semantic Ranking Engine
"""

from __future__ import annotations
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model Singleton ──────────────────────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

@dataclass
class RankResult:
    score: float
    verdict: str
    verdict_color: str
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    summary: str = ""

# ── Core AI Functions ────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    model = _get_model()
    # Normalizing makes Cosine Similarity identical to Dot Product
    return model.encode(text, normalize_embeddings=True)

def score_to_percent(similarity: float) -> float:
    """
    Calibrates raw similarity (usually 0.3-0.6) to a human-friendly scale (0-100).
    A similarity of 0.5 will now map to ~85% for a professional demo look.
    """
    raw_score = max(0.0, similarity)
    
    # Power-scaling formula to stretch the 'Top End' of the results
    # Anything above 0.45 similarity enters the 'High Match' territory
    calibrated = (raw_score ** 1.8) * 100 * 2.2 + 35
    
    # Cap at 98.5% (nothing is a perfect 100% in AI)
    final_score = min(98.5, calibrated)
    
    # If the match is truly poor (below 0.2 similarity), force a low score
    if raw_score < 0.2:
        final_score = raw_score * 100
        
    return round(final_score, 1)

def _extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """
    Extracts relevant technical terms while filtering noise words.
    """
    # Expanded stopwords to prevent 'about', 'daily', etc. from appearing
    noise_words = {
        "and", "the", "to", "of", "a", "in", "for", "is", "on", "with", "as", "by", 
        "at", "an", "be", "or", "are", "was", "were", "has", "have", "had", "not", 
        "this", "that", "it", "we", "you", "our", "your", "will", "can", "from", 
        "their", "they", "all", "more", "also", "who", "what", "when", "how", 
        "which", "than", "but", "if", "so", "up", "do", "any", "its", "into",
        "about", "daily", "used", "work", "role", "team", "build", "code", 
        "years", "using", "based", "working", "daily", "related", "plus"
    }
    
    # Regex to find technical terms (including C++, C#, .NET)
    tokens = re.findall(r"\b[a-z][a-z\+\#\.]{1,}\b", text.lower())
    counts: dict[str, int] = {}
    for tok in tokens:
        if tok not in noise_words and len(tok) > 2:
            counts[tok] = counts.get(tok, 0) + 1
            
    sorted_kw = sorted(counts, key=lambda k: counts[k], reverse=True)
    return sorted_kw[:top_n]

def _build_summary_of_fit(resume_text: str, jd_text: str, score: float) -> tuple[list[str], list[str], str]:
    resume_kw = set(_extract_keywords(resume_text, top_n=40))
    jd_kw = set(_extract_keywords(jd_text, top_n=40))

    strengths = sorted(resume_kw & jd_kw)[:8]
    gaps = sorted(jd_kw - resume_kw)[:6]

    if score >= 80:
        verdict_prose = "Excellent semantic match. Experience is highly aligned."
    elif score >= 60:
        verdict_prose = "Solid alignment. Good overlap with core technical requirements."
    elif score >= 40:
        verdict_prose = "Moderate match. Significant upskilling in specific tools required."
    else:
        verdict_prose = "Low alignment. Candidate background does not match role intent."

    summary = f"{verdict_prose} strengths include {', '.join(strengths)}. Missing: {', '.join(gaps)}."
    return strengths, gaps, summary

def _score_to_verdict(score: float) -> tuple[str, str]:
    if score >= 80: return "Strong Match", "#4ade80"
    if score >= 65: return "Good Match", "#a3e635"
    if score >= 50: return "Moderate Match", "#facc15"
    return "Weak Alignment", "#f87171"

# ── Public API ───────────────────────────────────────────────────────────────

def rank_resume(resume_text: str, jd_text: str) -> RankResult:
    res_vec = embed(resume_text)
    jd_vec = embed(jd_text)
    
    similarity = float(np.dot(res_vec, jd_vec))
    score = score_to_percent(similarity)
    
    verdict, color = _score_to_verdict(score)
    strengths, gaps, summary = _build_summary_of_fit(resume_text, jd_text, score)

    return RankResult(
        score=score, verdict=verdict, verdict_color=color,
        strengths=strengths, gaps=gaps, summary=summary
    )

def rank_multiple(resumes: list[tuple[str, str]], jd_text: str) -> list[dict]:
    jd_vec = embed(jd_text)
    results = []
    for filename, text in resumes:
        res_vec = embed(text)
        sim = float(np.dot(res_vec, jd_vec))
        score = score_to_percent(sim)
        v, c = _score_to_verdict(score)
        s, g, sum_p = _build_summary_of_fit(text, jd_text, score)
        results.append({
            "filename": filename, "score": score, "verdict": v,
            "verdict_color": c, "strengths": s, "gaps": g, "summary": sum_p
        })
    return sorted(results, key=lambda r: r["score"], reverse=True)