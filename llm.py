"""Обгортка над Gemini API (google-genai)."""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Callable, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# Ліміти безкоштовного рівня рахуються окремо для кожної моделі, тому ланцюжок моделей
# множить загальну квоту. Першими — Flash Lite (15 RPM / 500 RPD), далі Flash (5 RPM / 20 RPD).
DEFAULT_CHAIN = ("gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.8-flash, gemini-3.7-flash, "
                 "gemini-3.6-flash, gemini-3.5-flash, gemini-3-flash")
MODEL = os.getenv("GEMINI_MODEL", DEFAULT_CHAIN.split(",")[0].strip())
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_CHAIN).split(",") if m.strip()]
RETRIES_PER_MODEL = 3
COOLDOWN_SEC = 60  # після 429 модель пропускається цей час
_cooldown: dict[str, float] = {}

T = TypeVar("T")


class ModelBusyError(RuntimeError):
    """Усі моделі тимчасово недоступні."""


@lru_cache(maxsize=1)
def _available() -> frozenset[str] | None:
    """ID моделей, доступних ключу (без префікса models/); None — якщо список отримати не вдалося."""
    try:
        return frozenset(m.name.removeprefix("models/") for m in _client().models.list())
    except Exception:  # noqa: BLE001
        return None


def _resolve(name: str) -> str | None:
    """Точний ID або його preview-/датована версія (напр. gemini-3.5-flash-lite-preview-09-2026)."""
    avail = _available()
    if avail is None or name in avail:
        return name
    variants = sorted(m for m in avail if re.fullmatch(re.escape(name) + r"(-preview.*|-\d{2,}.*|-latest)", m))
    return variants[0] if variants else None


def model_chain() -> list[str]:
    resolved = (_resolve(m) for m in dict.fromkeys([MODEL, *FALLBACK_MODELS]))
    return [m for m in dict.fromkeys(resolved) if m]


def _with_fallback(call: Callable[[str], T]) -> T:
    """Пробує моделі по черзі: 429 — одразу наступна (і пауза для цієї), 5xx — повтор з паузою."""
    last: Exception | None = None
    chain = model_chain()
    ready = [m for m in chain if _cooldown.get(m, 0) < time.time()] or chain
    for model in ready:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return call(model)
            except errors.APIError as e:
                last = e
                if e.code == 429:
                    _cooldown[model] = time.time() + COOLDOWN_SEC
                    break
                if e.code == 404:
                    break
                if e.code not in (500, 503, 504):
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


def reply(system: str, history: list[dict]) -> str:
    # Без стрімінгу: Gemini може надіслати 503 посеред стріму, коли повторити вже неможливо.
    # Відповіді короткі (1–3 речення), тож цілою відповіддю — надійніше.
    resp = _with_fallback(lambda model: _client().models.generate_content(
        model=model,
        contents=_to_contents(history),
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
    ))
    return (resp.text or "").strip() or "Вибач, не зміг сформулювати відповідь. Спробуй переказати ще раз іншими словами."


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
