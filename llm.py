"""Обгортка над Gemini API (google-genai)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterator

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


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
    stream = _client().models.generate_content_stream(
        model=MODEL,
        contents=_to_contents(history),
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def make_report(system: str, history: list[dict], report_prompt: str) -> Report:
    contents = _to_contents(history + [{"role": "user", "content": report_prompt}])
    resp = _client().models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0.1,
            response_mime_type="application/json", response_schema=Report,
        ),
    )
    return resp.parsed if isinstance(resp.parsed, Report) else Report.model_validate_json(resp.text)
