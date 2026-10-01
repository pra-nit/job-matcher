"""
Reusable Ollama client with robust JSON handling (spec sections 28-29).

Responsibilities:
    * ONE client instance for the whole run (never reload the model)
    * ``/no_think`` + ``think=False`` for qwen3 models (faster, cleaner JSON)
    * JSON extraction that survives markdown fences, ``<think>`` blocks and
      surrounding prose
    * bounded retries with error feedback
    * Pydantic validation with a ``None`` result on final failure — a bad
      job never crashes the pipeline, the caller falls back to deterministic
      extraction
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import QwenSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*(.*?)```", re.S | re.I)


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first plausible JSON object from an LLM response.

    Handles: markdown code fences, <think> reasoning blocks, leading prose,
    trailing prose, and unbalanced garbage (returns None).
    """
    if not text:
        return None
    cleaned = _THINK_TAG_RE.sub("", text)
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    while start != -1:
        candidate = _balanced_object(cleaned, start)
        if candidate is not None:
            return candidate
        start = cleaned.find("{", start + 1)
    return None


def _balanced_object(text: str, start: int) -> Optional[str]:
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json(text: str) -> Optional[dict]:
    """Best-effort json.loads of an LLM response; None when hopeless."""
    block = extract_json_block(text)
    if block is None:
        return None
    for attempt in (block, _fix_common_issues(block)):
        try:
            data = json.loads(attempt)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _fix_common_issues(block: str) -> str:
    """Repair the two classic small-model JSON mistakes."""
    # trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", block)
    # smart quotes used as JSON delimiters
    fixed = fixed.replace("“", '"').replace("”", '"').replace("’", "'")
    return fixed


class OllamaClient:
    """Thin, reusable wrapper around the ``ollama`` python package."""

    def __init__(self, settings: QwenSettings) -> None:
        self.settings = settings
        self._client: Any = None

    # -- lazy connection ------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import ollama  # lazy: allows the rest of the app to run without it
            except ImportError as exc:
                raise RuntimeError(
                    "The 'ollama' package is not installed. "
                    "Install requirements.txt or run with --offline."
                ) from exc
            self._client = ollama.Client(host=self.settings.base_url)
        return self._client

    def ping(self) -> bool:
        """True when the Ollama server answers and the model exists."""
        try:
            models = self.client.list()
            model_ids = []
            models_obj = getattr(models, "models", None)
            if models_obj is None and isinstance(models, dict):
                models_obj = models.get("models", [])
            for m in models_obj or []:
                name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                if name:
                    model_ids.append(name)
            if model_ids and not any(self.settings.model in m or m.startswith(self.settings.model + ":") for m in model_ids):
                logger.warning(
                    "Model '%s' not found on the Ollama server (available: %s). "
                    "Run: ollama pull %s",
                    self.settings.model, model_ids, self.settings.model,
                )
                return False
            return True
        except Exception as exc:
            logger.error("Ollama server not reachable at %s: %s", self.settings.base_url, exc)
            return False

    # -- chat helpers ---------------------------------------------------------
    def chat(self, system: str, user: str, no_think: bool = True) -> Optional[str]:
        """Raw chat call returning assistant text (None on failure)."""
        suffix = ""
        if no_think and self.settings.disable_thinking and "qwen3" in self.settings.model:
            suffix = " /no_think"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user + suffix},
        ]
        options = {
            "temperature": self.settings.temperature,
            "num_ctx": self.settings.num_ctx,
            "num_predict": self.settings.num_predict,
        }
        for attempt in range(self.settings.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.settings.model,
                    "messages": messages,
                    "format": "json",
                    "options": options,
                }
                if self.settings.disable_thinking:
                    kwargs["think"] = False
                try:
                    response = self.client.chat(**kwargs)
                except TypeError:
                    # older ollama versions do not accept think=
                    kwargs.pop("think", None)
                    response = self.client.chat(**kwargs)
                return self._content(response)
            except Exception as exc:
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt + 1, self.settings.max_retries + 1, exc,
                )
                if attempt < self.settings.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        return None

    @staticmethod
    def _content(response: Any) -> Optional[str]:
        message = None
        if isinstance(response, dict):
            message = response.get("message") or {}
        else:
            message = getattr(response, "message", None)
        if message is None:
            return None
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not content:
            thinking = getattr(message, "thinking", None)
            if thinking:
                logger.debug("Model returned only thinking content; discarding")
            return None
        return str(content)

    def chat_json(
        self,
        system: str,
        user: str,
        schema_model: Optional[Type[T]] = None,
    ) -> Optional[T]:
        """
        Chat expecting strict JSON; validate against a Pydantic model.

        Retries once with corrective feedback when the model returns invalid
        JSON.  Returns a validated model instance, or None so the caller can
        use its deterministic fallback.
        """
        content = self.chat(system, user)
        if content is None:
            return None
        data = parse_llm_json(content)
        if data is None:
            logger.debug("Retrying: model returned unparseable JSON: %.200s", content)
            content = self.chat(
                system,
                user + "\n\nIMPORTANT: your previous answer was not valid JSON. "
                "Respond with a single valid JSON object and nothing else.",
            )
            if content is None:
                return None
            data = parse_llm_json(content)
        if data is None:
            logger.warning("Model produced unparseable JSON after retry; using fallback")
            return None
        if schema_model is None:
            return data  # type: ignore[return-value]
        try:
            return schema_model.model_validate(data)
        except ValidationError as exc:
            logger.warning("JSON failed schema validation (%s); using fallback", str(exc).splitlines()[0])
            # one schema-guided retry
            schema_hint = json.dumps(schema_model.model_json_schema().get("properties", {}), indent=0)
            content = self.chat(
                system + "\nYour JSON must contain exactly these fields:\n" + schema_hint,
                user,
            )
            if content is None:
                return None
            data = parse_llm_json(content)
            if data is None:
                return None
            try:
                return schema_model.model_validate(data)
            except ValidationError:
                logger.warning("Schema-guided retry also failed; using fallback")
                return None
