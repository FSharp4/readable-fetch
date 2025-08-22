from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, ValidationError

from readable_fetch.url_to_markdown import fetch_and_markdownify
from readable_fetch.url_to_markdown_gpt import fetch_and_markdownify_gpt


app = FastAPI(
    title="Readable-Fetch MCP Server",
    version="0.1.0",
    description="FastAPI-based MCP-like server exposing fetch_and_markdownify and fetch_and_markdownify_gpt tools.",
)


# ----- Pydantic models -----


class FetchArgs(BaseModel):
    url: str = Field(..., description="The URL of the webpage to process.")


class FetchGptArgs(BaseModel):
    url: str = Field(..., description="The URL of the webpage to process.")
    api_key: Optional[str] = Field(
        default=None,
        description="Explicit OpenAI API key. If omitted, the server will try OPENAI_SHARING_KEY then OPENAI_API_KEY environment variables.",
    )
    model: Optional[str] = Field(
        default="gpt-5-mini",
        description="OpenAI model to use for LLM-based extraction.",
    )
    chunk_chars: Optional[int] = Field(
        default=60000,
        description="Max characters per chunk for large HTML pages.",
    )
    token_threshold: Optional[int] = Field(
        default=60000,
        description="Max tokens allowed before chunking. If None, always chunk by size.",
    )
    prefer_sharing_key: Optional[bool] = Field(
        default=True,
        description="Prefer OPENAI_SHARING_KEY over OPENAI_API_KEY when resolving keys.",
    )
    preclean: Optional[bool] = Field(
        default=True,
        description="If True, remove script/style/noscript and HTML comments before processing.",
    )


class ToolInvokeRequest(BaseModel):
    tool: str = Field(..., description="Tool to invoke.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")


class ToolInvokeResponse(BaseModel):
    ok: bool
    result: Optional[str] = None
    error: Optional[str] = None


# ----- Endpoints -----


@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {"name": "readable-fetch-mcp", "version": "0.1.0"}


@app.get("/tools", tags=["discovery"])
def get_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "fetch_and_markdownify",
                "description": "Fetch URL and convert main content to Markdown (deterministic fallback via readability + markdownify).",
                "params": {
                    "url": {"type": "string", "required": True, "description": "Target URL"}
                },
            },
            {
                "name": "fetch_and_markdownify_gpt",
                "description": "LLM-based extraction via OpenAI; falls back to local conversion if keys are missing or model fails.",
                "params": {
                    "url": {"type": "string", "required": True},
                    "api_key": {"type": "string", "required": False},
                    "model": {"type": "string", "required": False, "default": "gpt-5-mini"},
                    "chunk_chars": {"type": "integer", "required": False, "default": 60000},
                    "token_threshold": {"type": ["integer", "null"], "required": False, "default": 60000},
                    "prefer_sharing_key": {"type": "boolean", "required": False, "default": True},
                    "preclean": {"type": "boolean", "required": False, "default": True},
                },
            },
        ]
    }


@app.post("/invoke", response_model=ToolInvokeResponse, tags=["invoke"])
def invoke(request: ToolInvokeRequest) -> ToolInvokeResponse:
    # Validate args via Pydantic models so missing/invalid args result in 422 responses.
    if request.tool == "fetch_and_markdownify":
        try:
            args = FetchArgs(**request.args)
        except ValidationError as e:
            # Return 422 to mirror FastAPI's validation response for missing/invalid args
            raise HTTPException(status_code=422, detail=e.errors())
        md = fetch_and_markdownify(args.url)
        return ToolInvokeResponse(ok=True, result=md)

    if request.tool == "fetch_and_markdownify_gpt":
        try:
            args = FetchGptArgs(**request.args)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        try:
            md = fetch_and_markdownify_gpt(
                url=args.url,
                api_key=args.api_key,
                model=args.model or "gpt-5-mini",
                chunk_chars=args.chunk_chars or 60000,
                token_threshold=args.token_threshold,
                prefer_sharing_key=args.prefer_sharing_key if args.prefer_sharing_key is not None else True,
                preclean=args.preclean if args.preclean is not None else True,
            )
            return ToolInvokeResponse(ok=True, result=md)
        except Exception as e:
            return ToolInvokeResponse(ok=False, error=f"{type(e).__name__}: {e}")

    # Unknown tool — return structured error (200 with ok=false) so callers get a consistent payload.
    return ToolInvokeResponse(ok=False, error=f"Unknown tool: {request.tool}")


# Convenience: allow `python -m src.mcp_server.server` to run uvicorn
def main() -> None:
    import uvicorn

    # Use the package-qualified module path so the server can be started
    # whether the repo is installed or run from the project root.
    uvicorn.run("readable_fetch.mcp_server.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
