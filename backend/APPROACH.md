Approach Document: TalentRank AI
Technical Challenge: Smart Talent Selection Engine

1. Solution Design & Architecture
   TalentRank AI is built on a decoupled Full-Stack architecture designed for high performance and modularity.

    Backend (FastAPI): Chosen for its asynchronous capabilities and native support for Pydantic data validation. It acts as the orchestration layer for the Machine Learning models.

    Frontend (Next.js 14): Utilizes React's "Client Components" for real-time UI updates and Tailwind CSS for a professional, recruiter-centric dashboard.

    Communication: Standardized RESTful API endpoints with CORS middleware to facilitate secure cross-origin resource sharing.

2. Technical Choices & Rationales
    A. Layout-Aware Parsing (The "Two-Column" Fix)
    Choice: IBM Docling.
    Rationale: Standard PDF parsers (like PyPDF2) extract text based on the stream order, which causes "text interleaving" in multi-column resumes. Docling uses a deep-learning pipeline to detect document regions (headings, sidebars, tables). This preserves the logical reading order, ensuring that the AI evaluates experience in context rather than as a scrambled "word soup."

    B. Semantic Vector Intelligence
    Choice: all-MiniLM-L6-v2 Transformer Model via Sentence-Transformers.
    Rationale: Traditional Applicant Tracking Systems (ATS) rely on exact keyword matching. Our approach maps both the Job Description (JD) and the Resume into a high-dimensional (384-dim) vector space.

    Cosine Similarity: We calculate the angular distance between these vectors. This allows the system to recognize that "NLP Developer" and "Language Model Engineer" are nearly identical, even if the specific words don't match.

    Calibration: We implemented a power-scaling algorithm to normalize raw cosine scores into human-readable compatibility percentages (0-100%).

    C. HR Intelligence & Question Generation
    Rationale: To move beyond simple filtering, the engine performs a "Keyword Diff" between the JD vector neighborhood and the Resume vector. It identifies specific technical gaps and uses template-based logic to generate targeted interview questions, grounded in the candidate's actual projects.

3. Implementation Workflow
    Ingestion: User uploads PDF/DOCX.

    Structuring: Docling converts raw bytes into structured Markdown.

    Embedding: Text is chunked and passed through the Transformer model.

    Ranking: Batch processing calculates similarity against the JD vector for all candidates simultaneously.

    Visualization: Results are rendered through a dynamic Fit-Score ring and categorical breakdowns.

4. Scalability & Future Improvements
    Given more time, the system could be enhanced with:

    Vector Database (ChromaDB): To store and index thousands of resumes for persistent, lightning-fast retrieval.

    Agentic AI (AWS Bedrock): Integrating an LLM agent to provide qualitative "Reasoning of Fit" and automated candidate outreach.

    OCR Integration: Enabling full support for scanned image-based resumes.