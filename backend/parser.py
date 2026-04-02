"""
parser.py — Layout-Aware Resume Parser using Docling (IBM)

Why Docling over PyPDF2/pdfminer?
  • Docling uses a deep-learning pipeline (DocLayNet model) that detects
    document regions — columns, tables, headers, figures — before extracting
    text. This prevents the classic two-column bug where left + right column
    words are interleaved on the same line.
  • Output is a structured DoclingDocument with ordered TextItem blocks,
    so we reconstruct reading order correctly.
"""

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


def _build_converter() -> DocumentConverter:
    """
    Build a DocumentConverter with layout-detection enabled.
    
    PdfPipelineOptions lets us toggle:
      - do_table_structure   : parse tables as grids (not raw text)
      - do_ocr               : run OCR on scanned/image-based PDFs
      - generate_page_images : skip — we don't need screenshots
    """
    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,   # Capture tables correctly
        do_ocr=False,              # Set True for scanned PDFs
        generate_page_images=False,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )


# Module-level singleton — converter loads ML models once and reuses them
_converter = _build_converter()


def parse_resume(file_path: str | Path) -> str:
    """
    Extract clean, layout-ordered text from a PDF or DOCX resume.

    Two-column handling:
      Docling's DocLayNet model classifies each page region as one of:
      [Text | Title | List | Table | Figure | Caption | Header | Footer].
      Regions are then ordered top-to-bottom, left column before right column,
      matching natural reading flow — completely avoiding the interleave problem.

    Args:
        file_path: Absolute or relative path to the resume file.

    Returns:
        A single clean string with sections separated by double newlines.

    Raises:
        FileNotFoundError : If the file does not exist.
        ValueError        : If the format is unsupported.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    supported = {".pdf", ".docx", ".doc"}
    if file_path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported format '{file_path.suffix}'. "
            f"Supported: {supported}"
        )

    # ── Convert ────────────────────────────────────────────────────────────────
    result = _converter.convert(str(file_path))
    doc = result.document  # DoclingDocument

    # ── Reconstruct text in reading order ─────────────────────────────────────
    # doc.export_to_markdown() gives the cleanest linearisation:
    #   - Headings become "## Section"
    #   - Tables become markdown tables (easy for the LLM to read)
    #   - Two-column content is already correctly ordered
    markdown_text: str = doc.export_to_markdown()

    # ── Light post-processing ──────────────────────────────────────────────────
    # Remove markdown syntax so the ranker gets plain prose
    clean_lines = []
    for line in markdown_text.splitlines():
        # Strip markdown heading markers  (##, ###, etc.)
        stripped = line.lstrip("#").strip()
        if stripped:
            clean_lines.append(stripped)

    return "\n\n".join(clean_lines)


# ── Quick smoke-test (run: python parser.py <path_to_pdf>) ────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <resume.pdf>")
        sys.exit(1)

    text = parse_resume(sys.argv[1])
    print("=" * 60)
    print(text[:2000])          # Preview first 2000 chars
    print("=" * 60)
    print(f"\nTotal characters extracted: {len(text)}")