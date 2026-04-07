# TalentRank AI — Smart Talent Selection Engine

TalentRank AI is a next-generation Applicant Tracking System (ATS) that moves beyond primitive keyword matching. By utilizing layout-aware document parsing and semantic vector similarity, it understands the intent behind a resume, ensuring qualified candidates are never overlooked due to synonymous terminology or complex formatting.

● The Problem
Traditional ATS platforms rely on rigid keyword matching, causing recruiters to accidentally reject highly qualified candidates who use equivalent but different terminology. Furthermore, standard parsers often fail on modern, non-linear, or two-column resume layouts, leading to "text interleaving" that makes resumes unreadable for AI. With recruiters spending an average of 6 seconds per resume, these technical limitations lead to missed talent and hiring fatigue.


● The Solution
TalentRank AI solves these pain points through a three-pillared approach:

1.Layout-Aware Ingestion: Uses Docling (IBM) to detect document regions (columns, tables, headers).    This ensures that two-column resumes are parsed in the correct reading order, preserving the context of the candidate's experience.
2.Semantic Ranking: Instead of counting words, the engine encodes resumes and Job Descriptions into 384-dimensional vector embeddings using `all-MiniLM-L6-v2`. We calculate Cosine Similarity to find the "mathematical" fit between a candidate and a role.
3.HR Intelligence:
    Analyze: Instant breakdown of strengths and semantic gaps.
    Rank: Bulk-upload up to 20 resumes to get a logically ranked shortlist.
    Questions: Automatically generates targeted interview questions grounded in the candidate's actual resume content to assist HR in the interview process.


● Tech Stack
Programming Languages: Python 3.10+, TypeScript
Backend: FastAPI (Uvicorn)
Frontend: React (Next.js), Tailwind CSS, Lucide React
AI/ML Libraries: `Docling` (Advanced Document Parsing)
     `Sentence-Transformers` (Vector Embeddings)
     `NumPy` (Vector Math & Cosine Similarity)
Database: Local Vector Storage (ChromaDB architecture)


● Setup Instructions

1. Prerequisites
 Python 3.10 or higher installed.
 Node.js (v18+) installed.

2. Backend Setup (Windows)
 cd backend

Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

Install dependencies
pip install fastapi uvicorn docling sentence-transformers numpy python-multipart

Start the server
python main.py