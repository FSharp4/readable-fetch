#!/usr/bin/env python3
"""
demo_gpt.py - Demonstrate fetch_and_markdownify_gpt from src/readable_fetch/url_to_markdown_gpt.py

Usage examples:
    python demo_gpt.py "https://example.com/article" --output article.md
    python demo_gpt.py "https://example.com/article" --token-threshold 30000 --no-preclean

Notes:
- The script will prefer OPENAI_SHARING_KEY then OPENAI_API_KEY if no --api-key is provided.
- Ensure dependencies installed with: uv add openai tiktoken
"""

import argparse
import os
import sys
# from typing import Optional

# Ensure 'src' directory is on sys.path so we can import the package module
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from readable_fetch.url_to_markdown_gpt import (
        estimate_tokens,
        split_raw_html,
        fetch_and_markdownify_gpt,
        get_api_key,
    )
except Exception as e:
    print("Error importing module 'readable_fetch.url_to_markdown_gpt':", e)
    print("Make sure you ran `uv add openai tiktoken` and that the 'src' path is correct.")
    raise

def parse_args():
    p = argparse.ArgumentParser(description="Demo: Convert a URL to Markdown using gpt-5-mini")
    p.add_argument("url", nargs="?", default="https://en.wikipedia.org/wiki/OpenAI", help="URL to convert (default: %(default)s)")
    p.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use (default: gpt-5-mini)")
    p.add_argument("--chunk-chars", type=int, default=60000, help="Character chunk size (default: 60000)")
    p.add_argument("--token-threshold", type=int, default=60000, help="Token threshold to gate single-call vs chunking (default: 60000). Set to 0 or 1 to force chunking.")
    p.add_argument("--no-preclean", dest="preclean", action="store_false", help="Disable minimal HTML pre-cleaning (script/style removal)")
    p.add_argument("--no-prefer-sharing-key", dest="prefer_sharing_key", action="store_false", help="Do not prefer OPENAI_SHARING_KEY over OPENAI_API_KEY")
    p.add_argument("--api-key", dest="api_key", default=None, help="Explicit OpenAI API key (overrides env vars)")
    p.add_argument("--output", "-o", dest="output", default="demo_article.md", help="Write Markdown to this file (default: %(default)s)")
    return p.parse_args()

def main():
    args = parse_args()

    # Resolve API key for display only (fetch_and_markdownify_gpt will also resolve)
    try:
        get_api_key(prefer_sharing_key=args.prefer_sharing_key, explicit=args.api_key)
        key_display = "found"
    except Exception:
        key_display = "none (will use local fallback)"

    print(f"URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Chunk chars: {args.chunk_chars}")
    print(f"Token threshold: {args.token_threshold}")
    print(f"Preclean: {args.preclean}")
    print(f"API key: {key_display}")
    print("Estimating token count (this may take a moment)...")

    # Quick fetch to estimate tokens and show chunking plan
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(args.url, headers=headers, timeout=10)
        r.raise_for_status()
        raw_html = r.text
    except Exception as e:
        print("Failed to fetch URL for estimation:", e)
        return

    # Minimal local pre-cleaning for the estimate (same as module)
    from readable_fetch.url_to_markdown_gpt import minimal_html_clean
    html_input = minimal_html_clean(raw_html) if args.preclean else raw_html

    tokens = estimate_tokens(html_input, model=args.model)
    print(f"Estimated tokens: {tokens}")

    will_chunk = False
    if args.token_threshold is not None and tokens > args.token_threshold:
        will_chunk = True
    if args.token_threshold is None:
        # token_threshold None means always chunk by size
        will_chunk = True

    print(f"Will chunk: {will_chunk}")

    if will_chunk:
        parts = split_raw_html(html_input, chunk_chars=args.chunk_chars)
        print(f"Split into {len(parts)} chunk(s) (approx sizes: {', '.join(str(len(p)) for p in parts[:5])}{'...' if len(parts)>5 else ''})")
    else:
        print("Single-shot conversion planned (no chunking).")

    print("Calling fetch_and_markdownify_gpt (this will call the OpenAI API)...")
    md = fetch_and_markdownify_gpt(
        url=args.url,
        api_key=args.api_key,
        model=args.model,
        chunk_chars=args.chunk_chars,
        token_threshold=args.token_threshold,
        prefer_sharing_key=args.prefer_sharing_key,
        preclean=args.preclean,
    )

    # if not isinstance(md, str):
    #     print("Unexpected non-string result from conversion.")
    #     return

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Markdown written to {args.output}")
        except Exception as e:
            print("Failed to write output file:", e)
    else:
        print("\n===== BEGIN MARKDOWN OUTPUT =====\n")
        print(md)
        print("\n===== END MARKDOWN OUTPUT =====\n")

if __name__ == "__main__":
    main()
