"""
main.py — Smart Talent Selection Engine · FastAPI Backend

Endpoints:
  POST /analyze            — Upload a single resume PDF/DOCX; returns score + analysis.
  POST /rank               — Upload multiple resumes against a JD; returns ranked list.
  POST /generate-questions — Upload resume + optional JD; returns categorised HR questions.
  GET  /health             — Liveness check.
  GET  /sample-jd          — Returns the built-in sample JD (handy for the demo).
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parser import parse_resume
from ranker import rank_multiple, rank_resume
from question_generator import generate_interview_questions

# ══════════════════════════════════════════════════════════════════════════════
# App Setup
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Smart Talent Selection Engine",
    description=(
        "AI-powered resume analysis system using layout-aware parsing (Docling) "
        "and semantic similarity ranking (Sentence-Transformers)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# Built-in Sample JD (used when caller does not supply one)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_JD = """
Senior Machine Learning Engineer

About the Role:
We are looking for a Senior ML Engineer to join our AI Platform team.
You will design, build, and deploy production-grade machine learning systems
that power our recommendation engine and NLP products used by millions daily.

Responsibilities:
- Design and train deep learning models using PyTorch or TensorFlow.
- Build and maintain scalable ML pipelines using Apache Airflow and Spark.
- Serve models via REST APIs built with FastAPI and containerised with Docker.
- Collaborate with data scientists to move experiments from notebook to production.
- Optimise model inference latency and throughput on GPU clusters (Kubernetes).
- Write clean, tested Python code and conduct peer code reviews.

Requirements:
- 3+ years of professional experience in machine learning engineering.
- Strong proficiency in Python; familiarity with C++ is a bonus.
- Deep understanding of NLP techniques: transformers, embeddings, fine-tuning.
- Experience with MLOps tools: MLflow, Weights & Biases, or similar.
- Solid understanding of SQL and experience with cloud platforms (AWS / GCP).
- Bachelor's or Master's degree in Computer Science, Statistics, or related field.

Nice to Have:
- Contributions to open-source ML projects.
- Experience with large language models (LLMs) and prompt engineering.
- Familiarity with vector databases (Pinecone, ChromaDB, Weaviate).
"""


# ══════════════════════════════════════════════════════════════════════════════
# Response Models
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeResponse(BaseModel):
    filename: str
    extracted_text_preview: str      # First 500 chars — full text can be large
    character_count: int
    score: float
    verdict: str
    verdict_color: str
    strengths: list[str]
    gaps: list[str]
    summary: str


class RankItem(BaseModel):
    rank: int
    filename: str
    score: float
    verdict: str
    verdict_color: str
    strengths: list[str]
    gaps: list[str]
    summary: str


class RankResponse(BaseModel):
    jd_preview: str
    total_candidates: int
    results: list[RankItem]


class HealthResponse(BaseModel):
    status: str
    version: str


class QuestionCategory(BaseModel):
    category: str
    icon: str
    questions: list[str]


class QuestionResponse(BaseModel):
    filename: str
    candidate_name: str
    total_questions: int
    categories: list[QuestionCategory]


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc"}


def _save_upload(upload: UploadFile) -> Path:
    """
    Save an UploadFile to a secure temp file and return its Path.
    Caller is responsible for deleting the file after use.
    """
    suffix = Path(upload.filename or "resume").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{suffix}' not supported. Use PDF or DOCX.",
        )
    # NamedTemporaryFile with delete=False so we can pass the path to Docling
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Liveness check — confirm the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/sample-jd", tags=["Utility"])
def get_sample_jd():
    """Return the built-in sample Job Description."""
    return {"jd": SAMPLE_JD.strip()}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_resume(
    resume: Annotated[UploadFile, File(description="PDF or DOCX resume file")],
    jd: Annotated[
        str,
        Form(description="Job Description text. Leave blank to use built-in sample JD."),
    ] = "",
):
    """
    **Single-resume analysis.**

    Upload one resume file. Optionally supply a custom JD in the form body.
    Returns:
    - Extracted text preview (first 500 chars)
    - Compatibility score (0–100%)
    - Verdict with colour tag
    - Matched strengths & potential gaps
    - Plain-English summary of fit
    """
    tmp_path: Path | None = None
    try:
        tmp_path = _save_upload(resume)
        resume_text = parse_resume(tmp_path)

        target_jd = jd.strip() if jd.strip() else SAMPLE_JD
        result = rank_resume(resume_text, target_jd)

        return AnalyzeResponse(
            filename=resume.filename or "unknown",
            extracted_text_preview=resume_text[:500],
            character_count=len(resume_text),
            score=result.score,
            verdict=result.verdict,
            verdict_color=result.verdict_color,
            strengths=result.strengths,
            gaps=result.gaps,
            summary=result.summary,
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        )
    finally:
        if tmp_path and tmp_path.exists():
            os.unlink(tmp_path)


@app.post("/rank", response_model=RankResponse, tags=["Analysis"])
async def rank_resumes(
    resumes: Annotated[
        list[UploadFile],
        File(description="One or more PDF/DOCX resume files"),
    ],
    jd: Annotated[
        str,
        Form(description="Job Description text. Leave blank for built-in sample JD."),
    ] = "",
):
    """
    **Multi-resume ranking.**

    Upload up to 20 resumes at once. They are all ranked against the
    provided (or built-in) JD and returned sorted by score, best first.

    Ideal for the hackathon demo: drop a folder of resumes and see instant rankings.
    """
    if len(resumes) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 resumes per request.",
        )

    target_jd = jd.strip() if jd.strip() else SAMPLE_JD
    tmp_paths: list[Path] = []
    resume_texts: list[tuple[str, str]] = []

    try:
        # ── Parse all uploads ──────────────────────────────────────────────
        for upload in resumes:
            tmp = _save_upload(upload)
            tmp_paths.append(tmp)
            try:
                text = parse_resume(tmp)
            except Exception as exc:
                text = f"[PARSE ERROR: {exc}]"
            resume_texts.append((upload.filename or "unknown", text))

        # ── Rank ───────────────────────────────────────────────────────────
        ranked = rank_multiple(resume_texts, target_jd)

        results = [
            RankItem(rank=i + 1, **item)
            for i, item in enumerate(ranked)
        ]

        return RankResponse(
            jd_preview=target_jd[:300] + ("..." if len(target_jd) > 300 else ""),
            total_candidates=len(results),
            results=results,
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking failed: {exc}",
        )
    finally:
        for p in tmp_paths:
            if p.exists():
                os.unlink(p)


@app.post("/generate-questions", response_model=QuestionResponse, tags=["Analysis"])
async def generate_questions(
    resume: Annotated[UploadFile, File(description="PDF or DOCX resume file")],
    jd: Annotated[
        str,
        Form(description="Job Description text. Leave blank for built-in sample JD."),
    ] = "",
):
    """
    **Interview Question Generator.**

    Analyses the resume and generates categorised interview questions for HR:
    - 🧠 Technical Skills (based on tools/languages found in resume)
    - 💼 Work Experience (based on roles, projects, internships)
    - 🎯 Role Fit (based on JD gaps and alignment)
    - 🌱 Behavioural (situational / soft skill questions)
    - 📚 Education & Projects (based on academic background)
    """
    tmp_path: Path | None = None
    try:
        tmp_path = _save_upload(resume)
        resume_text = parse_resume(tmp_path)
        target_jd = jd.strip() if jd.strip() else SAMPLE_JD
        result = generate_interview_questions(resume_text, target_jd)

        return QuestionResponse(
            filename=resume.filename or "unknown",
            candidate_name=result["candidate_name"],
            total_questions=result["total_questions"],
            categories=[QuestionCategory(**c) for c in result["categories"]],
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {exc}",
        )
    finally:
        if tmp_path and tmp_path.exists():
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Dev server entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)