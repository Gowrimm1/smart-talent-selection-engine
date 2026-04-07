"""
question_generator.py — AI-Powered Interview Question Generator

Strategy:
  1. Extract structured signals from the resume:
     - Candidate name
     - Technical skills / tools / languages
     - Work experience & roles
     - Projects
     - Education
     - Gaps vs the JD
  2. Map each signal to a question category.
  3. Generate specific, non-generic questions grounded in resume content.

No external LLM API needed — all logic is rule-based + template-driven,
making it fast, free, and offline-capable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Signal Extraction Helpers
# ══════════════════════════════════════════════════════════════════════════════

# Common technical skill keywords to detect
TECH_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "php", "ruby", "matlab",
    # ML / AI
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "huggingface",
    "transformers", "opencv", "nltk", "spacy", "pandas", "numpy", "matplotlib",
    "seaborn", "xgboost", "lightgbm", "bert", "gpt", "llm", "rag",
    "langchain", "embeddings", "chromadb", "pinecone", "weaviate",
    # Web / Backend
    "fastapi", "flask", "django", "express", "react", "nextjs", "vue",
    "angular", "nodejs", "rest", "graphql", "websocket",
    # Data / DB
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "firebase", "supabase",
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd", "github actions",
    "terraform", "ansible", "linux", "bash", "git",
    # MLOps
    "mlflow", "airflow", "spark", "hadoop", "databricks", "weights & biases",
    "wandb", "dvc", "bentoml", "triton",
}

EXPERIENCE_PATTERNS = [
    r"(?:software|ml|ai|data|backend|frontend|full.?stack|devops|research)\s+engineer",
    r"(?:data|research|ml|ai|nlp)\s+scientist",
    r"(?:senior|junior|lead|principal|staff)\s+\w+",
    r"intern(?:ship)?",
    r"developer",
    r"analyst",
    r"consultant",
    r"architect",
]

PROJECT_PATTERNS = [
    r"(?:built|developed|created|designed|implemented|deployed|trained|fine.?tuned)\s+(?:a|an|the)?\s*([A-Za-z\s\-]{3,40}?)(?:\s+using|\s+with|\s+for|[,\.\n])",
    r"project\s*[:\-]?\s*([A-Za-z\s\-]{3,40}?)(?:[,\.\n])",
]

EDUCATION_PATTERNS = [
    r"b\.?tech|bachelor|b\.?e\.?|b\.?sc",
    r"m\.?tech|master|m\.?e\.?|m\.?sc|mca",
    r"ph\.?d|doctorate",
    r"diploma",
]


def _extract_name(text: str) -> str:
    """Heuristic: first non-empty line that looks like a person's name."""
    for line in text.splitlines()[:10]:
        line = line.strip()
        # Name: 2-4 capitalized words, no digits, reasonable length
        if (
            2 <= len(line.split()) <= 4
            and re.match(r"^[A-Z][a-zA-Z\s\.]+$", line)
            and len(line) < 50
        ):
            return line
    return "the Candidate"


def _extract_skills(text: str) -> list[str]:
    """Return detected technical skills from the resume text."""
    lower = text.lower()
    found = []
    for skill in TECH_SKILLS:
        # Word-boundary match to avoid partial hits (e.g. "r" in "research")
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, lower):
            found.append(skill)
    return sorted(found)


def _extract_roles(text: str) -> list[str]:
    """Extract job titles / roles mentioned in the resume."""
    lower = text.lower()
    found = set()
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, lower):
            found.add(match.group(0).strip().title())
    return list(found)[:6]


def _extract_projects(text: str) -> list[str]:
    """Extract project names/descriptions from the resume."""
    projects = []
    for pattern in PROJECT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            proj = match.group(1).strip().strip(".,")
            if 3 < len(proj) < 60 and proj not in projects:
                projects.append(proj)
    return projects[:5]


def _extract_education(text: str) -> str:
    """Return highest detected education level."""
    lower = text.lower()
    if re.search(r"ph\.?d|doctorate", lower):
        return "PhD"
    if re.search(r"m\.?tech|master|m\.?e\.?|m\.?sc|mca", lower):
        return "Master's"
    if re.search(r"b\.?tech|bachelor|b\.?e\.?|b\.?sc", lower):
        return "Bachelor's"
    if re.search(r"diploma", lower):
        return "Diploma"
    return "Degree"


def _gap_skills(resume_text: str, jd_text: str) -> list[str]:
    """Return JD skills not found in the resume."""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    gaps = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, jd_lower) and not re.search(pattern, resume_lower):
            gaps.append(skill)
    return gaps[:6]


# ══════════════════════════════════════════════════════════════════════════════
# Question Templates
# ══════════════════════════════════════════════════════════════════════════════

def _technical_questions(skills: list[str], gaps: list[str]) -> list[str]:
    questions = []

    # Skill-specific deep dives
    skill_deep = {
        "python": "Can you walk us through a Python project where you had to optimise performance? What profiling tools did you use?",
        "pytorch": "How do you handle overfitting in PyTorch? Walk us through your regularisation strategy in a recent model.",
        "tensorflow": "Describe the difference between TensorFlow's Eager and Graph execution modes. When would you use each?",
        "docker": "Explain how you would containerise an ML model for production. What goes in your Dockerfile?",
        "kubernetes": "How does Kubernetes handle pod scheduling? Have you written custom resource limits for ML workloads?",
        "fastapi": "How do you handle async background tasks and request validation in FastAPI?",
        "sql": "Write a query to find the top 3 candidates per department by salary — can you explain your window function approach?",
        "react": "How do you manage global state in a large React app? Compare Context API vs Redux vs Zustand.",
        "aws": "Walk us through setting up an auto-scaling ML inference endpoint on AWS.",
        "mlflow": "How do you use MLflow to track experiments across different model architectures?",
        "git": "Describe your Git branching strategy. How do you handle merge conflicts in a team?",
        "bert": "How did you fine-tune BERT for a downstream task? What was your training setup and evaluation metric?",
        "langchain": "Describe a RAG pipeline you built using LangChain. What chunking strategy did you use?",
    }

    for skill in skills[:5]:
        if skill in skill_deep:
            questions.append(skill_deep[skill])
        else:
            questions.append(
                f"You listed {skill.upper()} on your resume — describe the most complex problem you solved using it."
            )

    # Gap-based challenge questions
    for gap in gaps[:3]:
        questions.append(
            f"This role requires strong {gap.upper()} skills. How quickly could you get up to speed, and what's your plan?"
        )

    return questions[:7]


def _experience_questions(roles: list[str], projects: list[str]) -> list[str]:
    questions = []

    if roles:
        for role in roles[:2]:
            questions.append(
                f"During your time as a {role}, what was the most technically challenging problem you solved end-to-end?"
            )
            questions.append(
                f"As a {role}, how did you balance technical debt against delivery timelines?"
            )

    if projects:
        for proj in projects[:3]:
            questions.append(
                f"Tell me about your '{proj}' project — what was the biggest technical decision you made, and why?"
            )
            questions.append(
                f"If you were to rebuild '{proj}' from scratch today, what would you do differently?"
            )

    # Generic fallback experience questions
    questions += [
        "Describe a time you had to debug a production issue under pressure. Walk us through your process.",
        "What's the most impactful project you've delivered? How did you measure its success?",
        "Have you ever had to refactor a large codebase? What was your strategy?",
    ]

    return questions[:6]


def _role_fit_questions(gaps: list[str], skills: list[str]) -> list[str]:
    questions = [
        "Why are you interested in this specific role, and how does it align with your 3-year career goal?",
        "What aspect of this job description excites you the most, and which part concerns you?",
        "How do your past projects directly prepare you for the responsibilities listed in this JD?",
    ]

    if gaps:
        gap_str = ", ".join(gaps[:3])
        questions.append(
            f"The JD emphasises {gap_str} — areas not prominently mentioned in your resume. "
            f"How do you plan to bridge this gap if hired?"
        )

    questions += [
        "Where do you see yourself in this role after 6 months? What would success look like?",
        "This team moves fast. Describe how you handle competing priorities and tight deadlines.",
    ]
    return questions[:5]


def _behavioural_questions() -> list[str]:
    return [
        "Tell me about a time you disagreed with a technical decision made by your team lead. How did you handle it?",
        "Describe a situation where you had to learn a completely new technology in a short timeframe. How did you approach it?",
        "Give an example of a time you received critical feedback on your code or work. What did you do with it?",
        "Tell me about a project that failed or didn't go as planned. What did you learn?",
        "Describe a time you had to collaborate with a non-technical stakeholder. How did you communicate complex ideas?",
        "Have you ever mentored a junior team member or peer? What was your approach?",
    ]


def _education_questions(education: str, skills: list[str]) -> list[str]:
    questions = [
        f"Your {education} degree — which coursework or thesis work is most relevant to this role?",
        "Did you do any research or academic projects? How did that shape your engineering approach?",
        "How do you stay current with rapidly changing tools and papers in this field?",
    ]
    if "bert" in skills or "pytorch" in skills or "tensorflow" in skills:
        questions.append(
            "Have you read or implemented any recent research papers? Walk us through one."
        )
    questions.append(
        "What's the most valuable thing you learned outside of your formal education?"
    )
    return questions[:4]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_interview_questions(resume_text: str, jd_text: str) -> dict:
    """
    Generate categorised interview questions from a resume and JD.

    Returns a dict with:
      candidate_name   : str
      total_questions  : int
      categories       : list of { category, icon, questions }
    """
    name = _extract_name(resume_text)
    skills = _extract_skills(resume_text)
    roles = _extract_roles(resume_text)
    projects = _extract_projects(resume_text)
    education = _extract_education(resume_text)
    gaps = _gap_skills(resume_text, jd_text)

    categories = [
        {
            "category": "Technical Skills",
            "icon": "🧠",
            "questions": _technical_questions(skills, gaps),
        },
        {
            "category": "Work Experience & Projects",
            "icon": "💼",
            "questions": _experience_questions(roles, projects),
        },
        {
            "category": "Role Fit & Motivation",
            "icon": "🎯",
            "questions": _role_fit_questions(gaps, skills),
        },
        {
            "category": "Behavioural & Soft Skills",
            "icon": "🌱",
            "questions": _behavioural_questions(),
        },
        {
            "category": "Education & Learning",
            "icon": "📚",
            "questions": _education_questions(education, skills),
        },
    ]

    total = sum(len(c["questions"]) for c in categories)

    return {
        "candidate_name": name,
        "total_questions": total,
        "categories": categories,
    }


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESUME = """
    Gowri M M
    AI/ML Engineer | Python Developer

    Skills: Python, TensorFlow, scikit-learn, FastAPI, SQL, Git, React
    Experience: ML Intern at TechCorp — built a sentiment analysis pipeline
    Projects: Built a Resume Classifier using BERT fine-tuning
    Education: B.Tech Computer Science, 2024, CGPA 8.5
    """
    JD = """
    ML Engineer — requires PyTorch, MLflow, Docker, Kubernetes, AWS, Airflow
    """
    result = generate_interview_questions(RESUME, JD)
    print(f"Candidate: {result['candidate_name']}")
    print(f"Total Questions: {result['total_questions']}\n")
    for cat in result["categories"]:
        print(f"{cat['icon']} {cat['category']}")
        for i, q in enumerate(cat["questions"], 1):
            print(f"  {i}. {q}")
        print()