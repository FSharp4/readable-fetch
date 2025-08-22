"""url_to_markdown_gpt.py: LLM-first extraction and Markdown conversion via OpenAI gpt-5-mini.

This module fetches the raw HTML for a URL and uses an LLM to:
- Extract the main article content (bypassing readability).
- Convert it to high-quality Markdown.
- Optionally chunk the HTML for very large pages and merge results with de-dup heuristics.

Environment variables:
- OPENAI_SHARING_KEY (preferred, if present)
- OPENAI_API_KEY (fallback)

Public function:
    fetch_and_markdownify_gpt(
        url: str,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        chunk_chars: int = 60000,
        token_threshold: Optional[int] = 60000,
        prefer_sharing_key: bool = True,
        preclean: bool = True
    ) -> str
"""

from __future__ import annotations

import os
import re
import time
import math
import html
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from markdownify import markdownify as md  # fallback conversion

# tiktoken is optional; used for better token estimation if installed
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False # pyright: ignore[reportConstantRedefinition]

# OpenAI Python SDK (2024+)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.3"
)

# Configure a module-level logger (no handlers by default)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_api_key(prefer_sharing_key: bool = True, explicit: Optional[str] = None) -> str:
    """
    Resolve the OpenAI API key to use.

    Order:
    - explicit (if provided)
    - OPENAI_SHARING_KEY (if prefer_sharing_key=True and set)
    - OPENAI_API_KEY
    """
    if explicit:
        return explicit

    if prefer_sharing_key:
        sharing = os.getenv("OPENAI_SHARING_KEY")
        if sharing:
            return sharing

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "No OpenAI key found. Set OPENAI_SHARING_KEY or OPENAI_API_KEY, "
        "or pass api_key explicitly."
    )


def estimate_tokens(text: str, model: str = "gpt-5-mini") -> int:
    """
    Estimate token count of text for the given model.

    Uses tiktoken if available; otherwise approximates ~4 chars/token.
    """
    if not text:
        return 0

    if _HAS_TIKTOKEN:
        try:
            # cl100k_base is used for GPT-4/3.5; use as best-effort
            enc = tiktoken.get_encoding("cl100k_base") # pyright: ignore[reportPossiblyUnboundVariable]
            return len(enc.encode(text))
        except Exception:
            pass

    # Fallback approximation
    return math.ceil(len(text) / 4)


def _remove_html_comments(s: str) -> str:
    return re.sub(r"<!--([\s\S]*?)-->", "", s)


def minimal_html_clean(raw_html: str) -> str:
    """
    Minimally clean raw HTML to reduce token waste while preserving content.

    - Remove <script>, <style>, and <noscript> contents
    - Remove HTML comments
    - Optionally trim excessive whitespace
    """
    if not raw_html:
        return raw_html

    s = raw_html
    # Remove script/style/noscript blocks
    s = re.sub(
        r"(?is)<(script|style|noscript)\b[\s\S]*?</\1\s*>", "", s, flags=re.IGNORECASE
    )
    # Remove comments
    s = _remove_html_comments(s)
    # Collapse very long runs of whitespace
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _extract_title_from_html(raw_html: str) -> Optional[str]:
    """
    Best-effort extraction of the <title> tag text from raw HTML.
    """
    try:
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
        if m:
            # Unescape HTML entities
            return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip())
    except Exception:
        pass
    return None


def _source_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def split_raw_html(html_text: str, chunk_chars: int = 60000) -> List[str]:
    """
    Split raw (pre-cleaned) HTML into chunks up to chunk_chars, preferring
    boundaries at structural tags (</h1-6>, </section>, </article>, </main>).
    If none found near the boundary, fallback to paragraph breaks or sentences.

    This is a character-based heuristic splitter to keep implementation simple.
    """
    chunks: List[str] = []
    n = len(html_text)
    i = 0

    # Precompile patterns
    structural_patterns = [
        re.compile(r"(?i)</h[1-6]\s*>"),
        re.compile(r"(?i)</section\s*>"),
        re.compile(r"(?i)</article\s*>"),
        re.compile(r"(?i)</main\s*>"),
    ]

    while i < n:
        end = min(i + chunk_chars, n)
        window = html_text[i:end]

        # Try to find the last structural boundary within this window
        cut = -1
        for pat in structural_patterns:
            for m in pat.finditer(window):
                cut = max(cut, m.end())

        if cut != -1:
            # Found a structural cutpoint
            chunks.append(window[:cut])
            i += cut
            continue

        # Try double newline as paragraph-ish boundary
        m_para = list(re.finditer(r"\n\s*\n", window))
        if m_para:
            cut = m_para[-1].end()
            chunks.append(window[:cut])
            i += cut
            continue

        # Try last period in the last ~1000 chars
        tail = window[max(0, len(window) - 1000) :]
        last_dot = tail.rfind(".")
        if last_dot != -1:
            cut = len(window) - len(tail) + last_dot + 1
            chunks.append(window[:cut])
            i += cut
            continue

        # Fallback: hard cut at window end
        chunks.append(window)
        i = end

    return chunks


def _normalize_heading_text(s: str) -> str:
    """
    Normalize heading text for deduplication comparisons:
    - lowercase
    - strip punctuation
    - collapse whitespace
    """
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


_HEADING_LINE_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def _first_heading_text(md: str) -> Optional[str]:
    """
    Get the text of the first heading in a Markdown string, if present.
    """
    m = _HEADING_LINE_RE.search(md)
    if not m:
        return None
    return m.group(2)


def _last_heading_text(md: str) -> Optional[str]:
    """
    Get the text of the last heading in a Markdown string, if present.
    """
    matches = list(_HEADING_LINE_RE.finditer(md))
    if not matches:
        return None
    return matches[-1].group(2)


def _strip_leading_first_heading(md: str) -> str:
    """
    Remove the first heading line if present at the very start (allowing whitespace).
    """
    # Find first non-empty line
    lines = md.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines):
        m = re.match(r"^\s{0,3}(#{1,6})\s+.*$", lines[idx])
        if m:
            # remove this heading line
            del lines[idx]
            # also remove any blank lines immediately following to avoid extra gaps
            while idx < len(lines) and lines[idx].strip() == "":
                del lines[idx]
            return "\n".join(lines).lstrip("\n")
    return md


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _trim_overlap(prev_md: str, next_md: str, min_overlap: int = 50) -> str:
    """
    Trim overlap where the end of prev_md repeats at the start of next_md.

    Compares case-insensitive, whitespace-collapsed suffix/prefix windows.
    If an overlap ≥ min_overlap is found among window sizes, it trims that
    overlapping prefix from next_md.
    """
    if not prev_md or not next_md:
        return next_md

    prev_tail_raw = prev_md[-200:]
    next_head_raw = next_md[:200]

    prev_tail = _collapse_ws(prev_tail_raw).lower()
    next_head = _collapse_ws(next_head_raw).lower()

    # Quick bail if no shared starting substring
    if not prev_tail or not next_head:
        return next_md

    # Try several window sizes
    for win in (200, 150, 120, 100, 80, 60, 50):
        # Map to raw slices proportionally (best-effort)
        # We don't map precisely; we test on collapsed strings for speed,
        # then approximate raw trimming by searching raw next_head_raw for the
        # matching collapsed suffix if possible. If not possible, trim nothing.
        if len(prev_tail) < win or len(next_head) < win:
            continue
        suf = prev_tail[-win:]
        pre = next_head[:win]
        if suf == pre and win >= min_overlap:
            # Find a raw substring in next_head_raw that collapses to pre.
            # Best-effort: try a direct case-insensitive raw match ignoring whitespace runs.
            # Build a loose regex that matches pre allowing variable whitespace.
            pattern = re.sub(r"\s+", r"\\s+", re.escape(next_head_raw[:len(next_head_raw)]), flags=re.IGNORECASE)
            # That approach may be too heavy; instead perform a simple heuristic:
            # Trim the first len(next_head_raw)//2 characters if overlap is detected.
            # This is a pragmatic balance to avoid complex mapping.
            approx_trim = max(50, len(next_head_raw) // 2)
            return next_md[approx_trim:].lstrip()

    return next_md


def merge_markdown_chunks(parts: List[str]) -> str:
    """
    Merge multiple Markdown parts while avoiding duplicated headings and
    trimming overlapping content at boundaries.
    """
    merged: str = ""
    last_heading_norm: Optional[str] = None

    for idx, part in enumerate(parts):
        if not part:
            continue
        part = part.lstrip("\ufeff")  # drop any BOM
        part = part.strip("\n")

        if not merged:
            merged = part.strip()
            last_h = _last_heading_text(merged)
            last_heading_norm = _normalize_heading_text(last_h) if last_h else None
            continue

        # Deduplicate repeated leading heading if equal to last heading of merged
        first_h = _first_heading_text(part)
        if first_h and last_heading_norm:
            if _normalize_heading_text(first_h) == last_heading_norm:
                part = _strip_leading_first_heading(part)

        # Trim overlaps at chunk boundary
        part = _trim_overlap(merged, part, min_overlap=50)

        # Ensure single blank line between sections
        merged = merged.rstrip() + "\n\n" + part.lstrip()

        # Update last heading
        last_h = _last_heading_text(merged)
        last_heading_norm = _normalize_heading_text(last_h) if last_h else None

    return merged.strip() + "\n"


def _build_system_prompt() -> str:
    return (
        "You are a professional content extractor and Markdown converter. "
        "Extract ONLY the primary article content from the provided raw HTML and convert it to clean, well-structured Markdown. "
        "Remove non-substantive content such as ads, navigation/menu items, cookie banners, subscription prompts, newsletter signups, social/share buttons, related-articles blocks, comment threads, footers, and repeated site boilerplate. "
        "Also remove non-informative filler sentences (for example: 'In this article we will...', 'Thanks for reading', promotional or conversational asides) unless they carry meaningful factual content. "
        "Preserve factual and substantive content: headings, paragraphs, lists, blockquotes, code blocks, inline code, links, and images (use ![alt](url)). "
        "Do not invent or hallucinate content. Output MUST be only Markdown and must contain no explanations, commentary, or extraneous markup."
    )


def _build_user_instructions(metadata: Dict[str, Any], part_index: int, part_total: int) -> str:
    rules = [
        "- Extract only the main article content. Ignore navigation, cookie banners, ads, related links, comments, headers, footers, and sidebars.",
        "- Preserve structure: headings (#), paragraphs, lists, blockquotes, code fences, inline code, links, and images as ![alt](url).",
        "- If possible, keep alt text for images; if unknown, use a brief description or the filename.",
        "- Preserve the original section order and hierarchy.",
        "- Do not hallucinate missing content.",
        "- Resolve relative URLs using the page URL if possible; otherwise leave as-is.",
    ]
    frontmatter = (
        "Include a YAML frontmatter block at the very top ONLY for the first part with fields:\n"
        "title, author, date, url, source_domain.\n"
        "If a field is unknown, omit it (do not invent values).\n"
    )

    part_info = ""
    if part_total > 1:
        part_info = (
            f"This is part {part_index} of {part_total}. "
            "Continue seamlessly from prior context without repetition. "
            "Do NOT repeat frontmatter except in part 1."
        )

    md = [
        f"Page URL: {metadata.get('url','')}",
        f"Source Domain: {metadata.get('source_domain','')}",
    ]
    if metadata.get("title"):
        md.append(f"Title (from <title>): {metadata['title']}")
    meta_block = "\n".join(md)

    return (
        f"{meta_block}\n\n"
        "Extraction and conversion rules:\n"
        + "\n".join(rules)
        + "\n\n"
        + frontmatter
        + ("\n" + part_info if part_info else "")
        + "\n\n"
        "Return ONLY Markdown.\n"
    )


def build_prompt(metadata: Dict[str, Any], chunk_html: str, part_index: int, part_total: int) -> Tuple[str, str]:
    """
    Build (system, user) prompts for the model.
    The user content includes instructions and the raw HTML chunk.
    """
    system = _build_system_prompt()
    instructions = _build_user_instructions(metadata, part_index, part_total)
    # Place instructions before the raw HTML to guide the model
    user = f"{instructions}\n\nRaw HTML:\n\n<html>\n{chunk_html}\n</html>\n"
    return system, user


def call_gpt_for_chunk(
    chunk_html: str,
    metadata: Dict[str, Any],
    model: str,
    api_key: str,
    part_index: int,
    part_total: int,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """
    Call OpenAI Chat Completions API to convert a single HTML chunk to Markdown.
    Retries on transient errors.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not available. Please ensure 'openai' is installed.")

    client = OpenAI(api_key=api_key)
    system, user = build_prompt(metadata, chunk_html, part_index, part_total)

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                timeout=timeout,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            # Retry on rate limit / server errors; otherwise re-raise
            msg = str(e).lower()
            transient = any(x in msg for x in ("rate limit", "timeout", "temporarily", "overloaded", "unavailable", "429", "500", "502", "503", "504"))
            if attempt < max_retries and transient:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            raise

    # Should not reach here
    return ""


def fetch_and_markdownify_gpt(
    url: str,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    chunk_chars: int = 60000,
    token_threshold: Optional[int] = 60000,
    prefer_sharing_key: bool = True,
    preclean: bool = True,
) -> str:
    """
    Fetch raw HTML from a URL and use OpenAI gpt-5-mini to extract and convert the main article to Markdown.

    Args:
        url: Target webpage URL.
        api_key: Explicit key; if None, resolve via OPENAI_SHARING_KEY then OPENAI_API_KEY.
        model: OpenAI model to use (default: "gpt-5-mini").
        chunk_chars: Character threshold for chunking raw HTML.
        token_threshold: Max tokens allowed before chunking; defaults to 60000. If None, always chunk by size.
        prefer_sharing_key: Prefer OPENAI_SHARING_KEY if available.
        preclean: If True, remove script/style/noscript and HTML comments before processing.

    Returns:
        Markdown as a string. On failures, attempts a deterministic local HTML-to-Markdown fallback.
    """
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        raw_html = resp.text
    except requests.exceptions.RequestException as e:
        return f"Error: Could not fetch the URL. Details: {e}"

    # Metadata
    metadata: Dict[str, Any] = {
        "url": url,
        "source_domain": _source_domain(url),
        "title": _extract_title_from_html(raw_html) or "",
    }

    # Pre-clean (still bypass readability)
    html_input = minimal_html_clean(raw_html) if preclean else raw_html

    # Token gating
    try:
        total_tokens = estimate_tokens(html_input, model=model)
    except Exception:
        total_tokens = estimate_tokens(html_input)

    try:
        resolved_key = get_api_key(prefer_sharing_key=prefer_sharing_key, explicit=api_key)
    except Exception as e:
        # If no key is available, fallback to local markdownify
        logger.warning("No OpenAI key available, using local markdownify fallback: %s", e)
        try:
            return md(html_input, heading_style="ATX")
        except Exception as e2:
            return f"Error: OpenAI key missing and local conversion failed: {e2}"

    try:
        # Decide single-shot vs chunked
        parts: List[str]
        if token_threshold is not None and total_tokens <= token_threshold:
            parts = [html_input]
        else:
            parts = split_raw_html(html_input, chunk_chars=chunk_chars)

        md_chunks: List[str] = []
        for idx, chunk in enumerate(parts, start=1):
            md_chunk = call_gpt_for_chunk(
                chunk_html=chunk,
                metadata=metadata,
                model=model,
                api_key=resolved_key,
                part_index=idx,
                part_total=len(parts),
            )
            if not md_chunk.strip():
                raise RuntimeError("Empty output from model.")
            md_chunks.append(md_chunk)

        if len(md_chunks) == 1:
            final_md = md_chunks[0].strip()
        else:
            final_md = merge_markdown_chunks(md_chunks)

        if not final_md.strip():
            raise RuntimeError("Model produced empty Markdown.")

        return final_md

    except Exception as e:
        # Fallback to deterministic local conversion
        logger.warning("Model conversion failed (%s). Falling back to local markdownify.", e)
        try:
            return md(html_input, heading_style="ATX")
        except Exception as e2:
            return f"Error: Model and local conversion failed: {e2}"
