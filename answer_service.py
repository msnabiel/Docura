"""Question answering over a built retrieval index."""

import json
import logging
import os
from pathlib import Path

import httpx
from google import genai
from google.genai import types

from config import generation_config
from retrieval import DocumentChunk, RetrievalIndex
from source_security import MAX_SOURCE_BYTES, validate_remote_url
from utils import clean_string_post


logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class AnswerService:
    def __init__(self, api_key: str | None = None, client=None,
                 model: str | None = None, fallback_model: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY_PAID")
        self.client = client or (genai.Client(api_key=api_key) if api_key else None)
        self.models = [model or os.getenv("DOCURA_GEMINI_MODEL", "gemini-3.7-flash")]
        fallback = fallback_model or os.getenv("DOCURA_GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
        if fallback and fallback not in self.models:
            self.models.append(fallback)
        config = {**generation_config, "system_instruction": (PROMPT_DIR / "system_prompt.txt").read_text()}
        self.generation_config = types.GenerateContentConfig(**config)
        self.prompt_template = (PROMPT_DIR / "phase1_prompt.txt").read_text()

    @staticmethod
    def _retrieve(index: RetrievalIndex, query: str, strategy: str) -> list[DocumentChunk]:
        if strategy == "semantic":
            results = index.semantic_search(query, top_k=8)
        elif strategy == "lexical":
            results = index.lexical_search(query, top_k=8)
        elif strategy == "hybrid":
            return index.hybrid_search(query, top_k=8)
        else:
            results = index.ensemble_search(query, top_k=16)
        return [result.chunk for result in results]

    @staticmethod
    def _context(chunks: list[DocumentChunk]) -> str:
        seen = set()
        sections = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            key = (source, chunk.text)
            if key not in seen:
                seen.add(key)
                sections.append(f"[Source: {source}]: {chunk.text}")
        return "\n\n".join(sections)

    async def _call_api(self, request: dict) -> str:
        try:
            method = request.get("type", "GET").upper()
            if method not in {"GET", "POST"}:
                return "Unsupported API method"
            url = validate_remote_url(request["url"])
            headers = request.get("headers") or []
            if isinstance(headers, list):
                headers = {item["key"]: item["value"] for item in headers}
            body = request.get("body") or {}
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                async with client.stream(method, url, headers=headers,
                                         params=body if method == "GET" else None,
                                         json=body if method == "POST" else None) as response:
                    if response.status_code != 200:
                        return f"API returned status {response.status_code}"
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_SOURCE_BYTES:
                            return "API response exceeded the size limit"
                    return bytes(content).decode("utf-8", errors="replace")
        except Exception as error:
            logger.warning("Model requested API call failed: %s", type(error).__name__)
            return "API call failed"

    async def ask(self, index: RetrievalIndex, question: str, strategy: str,
                  history: list) -> str:
        if self.client is None:
            raise RuntimeError("Gemini API key is not configured")
        recent_questions = [turn.content for turn in history if turn.role == "user"][-2:]
        retrieval_query = " ".join(recent_questions + [question])
        chunks = self._retrieve(index, retrieval_query, strategy)
        context = self._context(chunks)
        conversation = "\n".join(f"{turn.role}: {turn.content}" for turn in history)
        user_query = f"Previous conversation:\n{conversation}\n\nCurrent question: {question}" if conversation else question
        last_error = None
        for model_name in self.models:
            for attempt in range(2):
                api_results = []
                try:
                    for _ in range(3):
                        prompt = self.prompt_template.format(
                            query=user_query,
                            context=context,
                            api_response="\n".join(api_results) or "None",
                        )
                        response = await self.client.aio.models.generate_content(
                            model=model_name, contents=prompt, config=self.generation_config,
                        )
                        parsed = json.loads(response.text or "")
                        requested_api = parsed.get("need_api")
                        if isinstance(requested_api, dict):
                            api_results.append(await self._call_api(requested_api))
                            continue
                        answer = parsed.get("answer")
                        if isinstance(answer, str) and answer.strip():
                            return clean_string_post(answer)
                        raise ValueError("Model returned an empty answer")
                    raise ValueError("Model requested too many API calls")
                except Exception as error:
                    last_error = error
                    logger.warning("Answer attempt failed for %s: %s", model_name,
                                   type(error).__name__)
        raise RuntimeError("Answer generation failed") from last_error
