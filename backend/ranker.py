"""
ranker.py — Semantic Resume-to-JD Ranking Engine

Pipeline:
  1. Encode resume text + JD text → dense vector embeddings
     using sentence-transformers (all-MiniLM-L6-v2, 384-dim).
  2. Compute Cosine Similarity between the two vectors.
  3. Normalise to 0-100 compatibility score.
  4. Generate a structured "Summary of Fit" — strengths, gaps, verdict.

Why Cosine Similarity over dot-product?
  Cosine similarity measures the angle between vectors independent of
  their magnitudes, which matters here: a short resume and a long resume
  can both be "relevant" without the long one always winning.

Why all-MiniLM-L6-v2?
  • 22 MB model — fast inference, no GPU required.
  • Trained on 1B+ sentence pairs — strong semantic understanding.
  • 384 dimensions — compact yet expressive.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model (loaded once at import time) ────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class RankResult:
    """Everything the frontend needs to render a candidate card."""
    score: float                        # 0.0 – 100.0
    verdict: str                        # e.g. "Strong Match"
    verdict_color: str                  # Tailwind-friendly: "green" | "yellow" | "red"
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    summary: str = ""


# ── Core Functions ─────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """
    Convert text to a unit-normalised embedding vector.

    Args:
        text: Raw string (resume or JD).

    Returns:
        1-D numpy array of shape (384,), L2-normalised.
    """
    model = _get_model()
    # normalize_embeddings=True → cosine similarity == dot product (faster)
    vector = model.encode(text, normalize_embeddings=True)
    return vector


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two unit-normalised vectors.
    Returns a value in [-1, 1]. For semantic text, typically in [0, 1].
    """
    return float(np.dot(vec_a, vec_b))


def score_to_percent(similarity: float) -> float:
    """
    Map cosine similarity [-1, 1] → [0, 100] percentage.
    We clamp to [0, 1] first because negative similarity is meaningless here.
    """
    clamped = max(0.0, min(1.0, similarity))
    return round(clamped * 100, 1)


def _extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """
    Naive keyword extraction: lowercase tokens, filter stopwords, keep top_n
    by frequency. Used to surface readable strengths/gaps without an LLM.
    """
    stopwords = {
        "and", "the", "to", "of", "a", "in", "for", "is", "on", "with",
        "as", "by", "at", "an", "be", "or", "are", "was", "were", "has",
        "have", "had", "not", "this", "that", "it", "we", "you", "our",
        "your", "will", "can", "from", "their", "they", "all", "more",
        "also", "i", "my", "me", "who", "what", "when", "how", "which",
        "than", "but", "if", "so", "up", "do", "any", "its", "into",
    }
    tokens = re.findall(r"\b[a-z][a-z\+\#\.]{1,}\b", text.lower())
    counts: dict[str, int] = {}
    for tok in tokens:
        if tok not in stopwords:
            counts[tok] = counts.get(tok, 0) + 1
    sorted_kw = sorted(counts, key=lambda k: counts[k], reverse=True)
    return sorted_kw[:top_n]


def _build_summary_of_fit(
    resume_text: str,
    jd_text: str,
    score: float,
) -> tuple[list[str], list[str], str]:
    """
    Heuristically derive Strengths, Gaps, and a prose Summary of Fit.

    Returns:
        strengths : Keywords present in both resume and JD.
        gaps      : JD keywords absent from resume.
        summary   : One paragraph of human-readable assessment.
    """
    resume_kw = set(_extract_keywords(resume_text, top_n=40))
    jd_kw = set(_extract_keywords(jd_text, top_n=40))

    strengths = sorted(resume_kw & jd_kw)[:8]   # Overlap
    gaps = sorted(jd_kw - resume_kw)[:6]         # Missing from resume

    if score >= 80:
        verdict_sentence = (
            "This candidate is an excellent semantic match for the role. "
            "Their experience and skills are well-aligned with the job requirements."
        )
    elif score >= 60:
        verdict_sentence = (
            "This candidate shows a solid alignment with several core requirements. "
            "There are some gaps worth exploring in an interview."
        )
    elif score >= 40:
        verdict_sentence = (
            "The candidate has partial overlap with the role. "
            "Significant upskilling or experience gaps may exist."
        )
    else:
        verdict_sentence = (
            "Low semantic alignment detected. "
            "This resume does not closely match the job description."
        )

    strengths_prose = (
        f"Key overlapping skills/themes: {', '.join(strengths)}." if strengths
        else "No strong keyword overlap detected."
    )
    gaps_prose = (
        f"Potential missing areas: {', '.join(gaps)}." if gaps
        else "No significant gaps identified."
    )

    summary = f"{verdict_sentence} {strengths_prose} {gaps_prose}"
    return strengths, gaps, summary


def _score_to_verdict(score: float) -> tuple[str, str]:
    """Return a human-readable verdict and a color tag based on score."""
    if score >= 80:
        return "Strong Match", "green"
    elif score >= 65:
        return "Good Match", "lime"
    elif score >= 50:
        return "Moderate Match", "yellow"
    elif score >= 35:
        return "Weak Match", "orange"
    else:
        return "Poor Match", "red"


# ── Public API ─────────────────────────────────────────────────────────────────

def rank_resume(resume_text: str, jd_text: str) -> RankResult:
    """
    Main entry point. Score a single resume against a JD.

    Args:
        resume_text : Cleaned text extracted from the candidate's resume.
        jd_text     : Full text of the Job Description.

    Returns:
        RankResult dataclass with score, verdict, strengths, gaps, summary.
    """
    if not resume_text.strip():
        raise ValueError("Resume text is empty. Check the parser output.")
    if not jd_text.strip():
        raise ValueError("Job Description text is empty.")

    resume_vec = embed(resume_text)
    jd_vec = embed(jd_text)

    similarity = cosine_similarity(resume_vec, jd_vec)
    score = score_to_percent(similarity)

    verdict, color = _score_to_verdict(score)
    strengths, gaps, summary = _build_summary_of_fit(resume_text, jd_text, score)

    return RankResult(
        score=score,
        verdict=verdict,
        verdict_color=color,
        strengths=strengths,
        gaps=gaps,
        summary=summary,
    )


def rank_multiple(
    resumes: list[tuple[str, str]],   # [(filename, text), ...]
    jd_text: str,
) -> list[dict]:
    """
    Rank a list of resumes against a single JD.
    Encodes the JD only once for efficiency.

    Args:
        resumes : List of (filename, resume_text) tuples.
        jd_text : Full JD text.

    Returns:
        Sorted list of result dicts, best match first.
    """
    jd_vec = embed(jd_text)
    results = []

    for filename, resume_text in resumes:
        try:
            resume_vec = embed(resume_text)
            similarity = cosine_similarity(resume_vec, jd_vec)
            score = score_to_percent(similarity)
            verdict, color = _score_to_verdict(score)
            strengths, gaps, summary = _build_summary_of_fit(
                resume_text, jd_text, score
            )
            results.append({
                "filename": filename,
                "score": score,
                "verdict": verdict,
                "verdict_color": color,
                "strengths": strengths,
                "gaps": gaps,
                "summary": summary,
            })
        except Exception as exc:
            results.append({
                "filename": filename,
                "score": 0.0,
                "verdict": "Parse Error",
                "verdict_color": "red",
                "strengths": [],
                "gaps": [],
                "summary": str(exc),
            })

    return sorted(results, key=lambda r: r["score"], reverse=True)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_JD = """
    Senior Machine Learning Engineer
    We are looking for an ML engineer with 3+ years of experience in Python,
    deep learning frameworks (PyTorch or TensorFlow), NLP, and deploying models
    to production. Experience with Docker, Kubernetes, and FastAPI is a plus.
    You will design, train, and serve large-scale recommendation and NLP models.
    """

    SAMPLE_RESUME = """
    John Doe — Software Engineer
    Skills: Python, TensorFlow, scikit-learn, REST APIs, Flask, SQL, Git.
    Experience: 2 years at a fintech startup building fraud detection models.
    Education: B.Tech Computer Science, 2022.
    Projects: Sentiment analysis pipeline, image classification with CNNs.
    """

    result = rank_resume(SAMPLE_RESUME, SAMPLE_JD)
    print(f"Score      : {result.score}%")
    print(f"Verdict    : {result.verdict}")
    print(f"Strengths  : {result.strengths}")
    print(f"Gaps       : {result.gaps}")
    print(f"Summary    : {textwrap.fill(result.summary, 70)}")