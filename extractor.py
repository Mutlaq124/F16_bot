"""
PDF text extraction pipeline using MinerU (primary) or PyMuPDF (fallback).
Clean text output with post-processing and structured heading-based chunking.
"""

import re
import sys
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Boilerplate patterns to strip from F-16 manual text ─────────────────────────────────────────
F16_BOILERPLATE_PATTERNS = [
    # Running page header: raw PyMuPDF can give "TO 1F-16CM/AM-1 BMS \n2\n" or "BMS42" inline.
    # The [\s]* between BMS and \d+ handles both: collapsed ("BMS2") and split ("BMS \n2").
    r"TO\s+1F-16[A-Z0-9/\-]*\s+BMS\s*\n?\s*\d+",
    # Old multi-line header variant (belt-and-suspenders)
    r"TO 1F-16[A-Z0-9/\-]*\s*BMS[^\n]*\n\d+\n[^\n]*",
    r"CHANGE\s+\d+\.\d+\.\d+[^\n]*",                   # change tracking
    r"^T\.O\.\s+1F-16[^\n]*$",                          # T.O. references
    r"^UNCLASSIFIED\s*$",                                # classification markings
    r"^FOR OFFICIAL USE ONLY\s*$",
    r"\f",                                               # form feeds
    r"[ \t]+\n",                                         # trailing whitespace
    r"\n{4,}",                                           # 4+ consecutive blank lines -> 2
]

GENERAL_BOILERPLATE_PATTERNS = [
    r"^Page \d+ of \d+\s*$",
    r"^\d{1,4}\s*$",                                     # lone page number lines
    r"^[-─═]{5,}\s*$",                                   # pure separator lines
]


@dataclass
class TextChunk:
    heading: str
    content: str
    source_file: str
    chunk_index: int
    total_chunks: int

    def to_indexed_text(self) -> str:
        parts = [f"Source: {self.source_file}"]
        if self.heading:
            # Clean page markers: "=== Page 42 | F_16_manual.pdf ===" -> "Page 42"
            h = self.heading.strip()
            page_match = re.match(r"===\s*(Page\s*\d+)\s*\|[^=]*===", h)
            if page_match:
                h = page_match.group(1)
            else:
                h = h.strip("# ")
            parts.append(f"[Section: {h}]")
        parts.append(self.content.strip())
        return "\n\n".join(p for p in parts if p.strip())


def _fix_pymupdf_word_breaks(text: str) -> str:
    """
    PyMuPDF sometimes concatenates words at line/block boundaries with no space.
    Pattern: lowercase-letter immediately followed by Uppercase-letter (e.g. "ForewordThis").
    Insert a space between them.
    Also handles digit-immediately-into-uppercase: "50Compressor" -> "50 Compressor".
    """
    # Lowercase -> Uppercase (word boundary with no space)
    text = re.sub(r"([a-z,;:])([A-Z])", r"\1 \2", text)
    # Period/digit -> Uppercase (e.g. "RPM.The" or "50Compressor")
    text = re.sub(r"([.!?])(\s*)([A-Z])", lambda m: f"{m.group(1)} {m.group(3)}", text)
    # Digit -> Uppercase (section numbers running into headings)
    text = re.sub(r"(\d)([A-Z][a-z])", r"\1 \2", text)
    return text


def _clean_text(text: str, extra_patterns: List[str] = None) -> str:
    """Apply boilerplate removal patterns and normalize whitespace."""
    all_patterns = F16_BOILERPLATE_PATTERNS + GENERAL_BOILERPLATE_PATTERNS + (extra_patterns or [])
    for pattern in all_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _fix_pymupdf_word_breaks(text)
    return text.strip()


def _chunk_by_headings(
    text: str,
    source_file: str,
    min_chars: int = 150,
) -> List[TextChunk]:
    """
    Split text into semantic chunks based on:
      1. Markdown-style headings (# / ## / ###)  — MinerU output
      2. PyMuPDF page markers (=== Page N | file ===) — PyMuPDF fallback

    Each chunk = heading/marker + content below it.
    Falls back to paragraph splits if no structural markers are found.
    """
    # Combined heading pattern: markdown headings OR PyMuPDF page markers
    heading_re = re.compile(
        r"^(#{1,3}\s+.+|===\s*Page\s+\d+\s*\|[^=\n]*===)$",
        re.MULTILINE
    )
    parts = heading_re.split(text)

    chunks: List[TextChunk] = []
    current_heading = ""
    current_content_parts = []

    def flush(heading: str, content_parts: list):
        content = "\n".join(content_parts).strip()
        if not content:
            return
        # append to previous chunk if too short (less than min_chars)
        if len(content) < min_chars and chunks:
            chunks[-1].content += "\n\n" + content
        else:
            chunks.append(TextChunk(
                heading=heading,
                content=content,
                source_file=source_file,
                chunk_index=0,
                total_chunks=0,
            ))

    for part in parts:
        if heading_re.match(part):
            flush(current_heading, current_content_parts)
            current_heading = part
            current_content_parts = []
        else:
            current_content_parts.append(part)

    flush(current_heading, current_content_parts)

    if not chunks:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_chars]
        for p in paragraphs:
            chunks.append(TextChunk(
                heading="",
                content=p,
                source_file=source_file,
                chunk_index=0,
                total_chunks=0,
            ))

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i + 1
        chunk.total_chunks = len(chunks)


    return chunks


def _find_mineru_exe(name: str) -> Optional[str]:
    """
    Find a MinerU CLI executable.
    First checks the same Scripts/ dir as the running Python interpreter
    (handles conda envs where the Scripts dir may not be on subprocess PATH),
    then falls back to shutil.which.
    """
    # Derive Scripts dir from the current Python executable
    scripts_dir = Path(sys.executable).parent / "Scripts"
    candidate = scripts_dir / f"{name}.exe"  # Windows
    if candidate.exists():
        return str(candidate)
    candidate_no_ext = scripts_dir / name  # Linux/Mac
    if candidate_no_ext.exists():
        return str(candidate_no_ext)
    return shutil.which(name)  # fallback to PATH lookup


def _run_mineru_cmd(cmd: list, pdf_path: Path, tmp: str) -> Optional[str]:
    """Helper: run a MinerU variant command and return markdown text, or None."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            logger.debug(f"  MinerU cmd {cmd[0]} exited {result.returncode}: {result.stderr[:300]}")
            return None

        # Both magic-pdf v1 and mineru v2 write .md files somewhere under tmp.
        # v1: tmp/<stem>/auto/<stem>.md
        # v2: tmp/<stem>/<stem>.md  OR  tmp/auto/<stem>.md
        # rglob catches all cases.
        md_files = [
            p for p in Path(tmp).rglob("*.md")
            if p.stat().st_size > 100  # skip near-empty stubs
        ]
        if not md_files:
            logger.debug(f"  MinerU cmd {cmd[0]}: ran OK but no .md output found.")
            return None

        # Pick the largest .md file (the full-content one, not summary stubs)
        md_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        text = md_files[0].read_text(encoding="utf-8", errors="ignore")
        logger.debug(f"  MinerU cmd {cmd[0]}: using {md_files[0].name} ({len(text)} chars)")
        return text

    except FileNotFoundError:
        logger.debug(f"  MinerU cmd '{cmd[0]}' not found in PATH.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"  MinerU cmd '{cmd[0]}' timed out after 600s.")
        return None
    except Exception as e:
        logger.warning(f"  MinerU cmd '{cmd[0]}' failed: {e}")
        return None


def _extract_with_mineru(pdf_path: Path) -> Optional[str]:
    """
    Run MinerU CLI to extract PDF as markdown. Returns text or None.

    Tries in order:
      1. magic-pdf (v1.x CLI)   — magic-pdf -p <file> -o <dir> -m auto
      2. mineru    (v2.x CLI)   — mineru convert <file> --output <dir>
    Falls back to PyMuPDF if both fail.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # ── Attempt 1: magic-pdf (v1.x) ─────────────────────────────────
        magic_pdf_exe = _find_mineru_exe("magic-pdf")
        if magic_pdf_exe:
            text = _run_mineru_cmd(
                [magic_pdf_exe, "-p", str(pdf_path), "-o", tmp, "-m", "auto"],
                pdf_path, tmp
            )
            if text:
                return text
            logger.warning("  magic-pdf ran but produced no usable output.")

        # ── Attempt 2: mineru (v2.x) ────────────────────────────────────
        mineru_exe = _find_mineru_exe("mineru")
        if mineru_exe:
            text = _run_mineru_cmd(
                [mineru_exe, "convert", str(pdf_path), "--output", tmp],
                pdf_path, tmp
            )
            if text:
                return text
            logger.warning("  mineru (v2) ran but produced no usable output.")

        return None


def _extract_with_pymupdf(pdf_path: Path) -> Optional[str]:
    """Page-by-page text extraction via PyMuPDF with page markers."""
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not installed: pip install pymupdf")
        return None

    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            if text:
                pages.append(f"=== Page {i+1} | {pdf_path.name} ===\n{text}")
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return None


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Text file read failed for {path}: {e}")
        return None


def extract_document(path: Path, extra_clean_patterns: List[str] = None) -> List[TextChunk]:
    """
    Main entry point. Returns a list of semantic TextChunk objects ready for LightRAG insertion.

    Strategy:
        PDF  -> MinerU (if available) -> heading-based chunks
             -> PyMuPDF fallback       -> heading/page-based chunks
        TXT/MD -> direct read         -> heading-based chunks
    """
    path = Path(path)
    suffix = path.suffix.lower()
    logger.info(f"Extracting: {path.name}")

    if suffix == ".pdf":
        mineru_available = bool(_find_mineru_exe("magic-pdf") or _find_mineru_exe("mineru"))

        raw: Optional[str] = None
        if mineru_available:
            logger.info("  Trying MinerU for extraction (magic-pdf / mineru)")
            raw = _extract_with_mineru(path)
            if raw:
                logger.info(f"  MinerU: {len(raw)} chars extracted")
            else:
                logger.warning("  MinerU failed, falling back to PyMuPDF")

        if not raw:
            logger.info("  Using PyMuPDF for extraction")
            raw = _extract_with_pymupdf(path)

    elif suffix in (".txt", ".md"):
        raw = _read_text_file(path)
    else:
        logger.warning(f"  Unsupported file type: {suffix}")
        return []

    if not raw or not raw.strip():
        logger.error(f"  No content extracted from {path.name}")
        return []

    cleaned = _clean_text(raw, extra_patterns=extra_clean_patterns)
    chunks = _chunk_by_headings(cleaned, source_file=path.name)

    logger.info(f"  Post-clean: {len(cleaned)} chars -> {len(chunks)} semantic chunks")
    return chunks
