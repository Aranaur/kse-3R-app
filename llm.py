"""Обгортка над Gemini API (google-genai)."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Callable, Iterator, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Запасні моделі через кому — на випадок перевантаження (503) або вичерпаного ліміту (429) основної
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest").split(",") if m.strip()]
RETRIES_PER_MODEL = 3
_RETRYABLE = {429, 500, 503, 504}

T = TypeVar("T")


class ModelBusyError(RuntimeError):
    """Усі моделі тимчасово недоступні."""


def _with_fallback(call: Callable[[str], T]) -> T:
    """Повторює виклик з експоненційною паузою, потім переходить до запасних моделей."""
    last: Exception | None = None
    for model in dict.fromkeys([MODEL, *FALLBACK_MODELS]):
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return call(model)
            except errors.APIError as e:
                last = e
                if e.code == 404:  # модель недоступна — одразу наступна
                    break
                if e.code not in _RETRYABLE:
                    raise
                time.sleep(2 ** attempt)
    raise ModelBusyError(str(last)) from last


class Report(BaseModel):
    completeness_percent: int = Field(description="0–100, частка розкритих ключових ідей")
    accuracy: int = Field(description="1–5, фактична точність")
    independence: int = Field(description="1–5, самостійність відтворення")
    covered_points: list[str] = Field(description="Ключові ідеї, які учень розкрив сам")
    missed_points: list[str] = Field(description="Ключові ідеї, які учень не згадав")
    misconceptions: list[str] = Field(description="Фактичні помилки або хибні уявлення")
    strengths: str = Field(description="Що вдалося (1–2 речення)")
    recommendations: str = Field(description="Що повторити учню (1–2 речення)")
    teacher_summary: str = Field(description="Короткий висновок для викладача (2–3 речення)")


@lru_cache(maxsize=1)  # один клієнт на процес: інакше його закриває GC посеред стрімінгу
def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Не задано GEMINI_API_KEY у Secrets застосунку.")
    return genai.Client(api_key=key)


def _to_contents(history: list[dict]) -> list[types.Content]:
    return [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in history
    ]


def stream_reply(system: str, history: list[dict]) -> Iterator[str]:
    def start(model: str):
        stream = iter(_client().models.generate_content_stream(
            model=model,
            contents=_to_contents(history),
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
        ))
        # Помилки перевантаження приходять із першим чанком — тягнемо його тут, щоб їх можна було повторити
        return stream, next(stream, None)

    stream, first = _with_fallback(start)
    if first is not None and first.text:
        yield first.text
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def make_report(system: str, history: list[dict], report_prompt: str) -> Report:
    contents = _to_contents(history + [{"role": "user", "content": report_prompt}])
    resp = _with_fallback(lambda model: _client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0.1,
            response_mime_type="application/json", response_schema=Report,
        ),
    ))
    return resp.parsed if isinstance(resp.parsed, Report) else Report.model_validate_json(resp.text)
