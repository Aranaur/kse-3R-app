"""Зберігання завдань і звітів.

Локально все лежить у ./data. Якщо задано HF_TOKEN і DATA_REPO (приватний HF Dataset),
файли додатково синхронізуються туди — інакше на безкоштовному Space дані зникнуть
після перезапуску контейнера.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
BUILTIN_ASSIGNMENTS = BASE_DIR / "assignments"
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
ASSIGN_DIR = DATA_DIR / "assignments"
REPORT_DIR = DATA_DIR / "reports"

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_REPO = os.getenv("DATA_REPO")  # напр. "kse/3r-data"
_hub_enabled = bool(HF_TOKEN and DATA_REPO)

for d in (ASSIGN_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------- синхронізація з HF Dataset ----------

def pull_from_hub() -> None:
    """Завантажує всі дані з HF Dataset у локальну папку (викликається при старті)."""
    if not _hub_enabled:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATA_REPO, repo_type="dataset", token=HF_TOKEN,
        local_dir=DATA_DIR, allow_patterns=["assignments/*", "reports/*"],
    )


def _push(path: Path) -> None:
    if not _hub_enabled:
        return
    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=str(path), path_in_repo=path.relative_to(DATA_DIR).as_posix(),
        repo_id=DATA_REPO, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"add {path.name}",
    )


def hub_status() -> str:
    return f"HF Dataset `{DATA_REPO}`" if _hub_enabled else "локальна папка (дані зникнуть після перезапуску)"


# ---------- завдання ----------

def _parse_md(path: Path) -> dict:
    """Markdown з YAML front matter: --- meta --- текст параграфа."""
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    meta.setdefault("id", path.stem)
    meta.setdefault("title", path.stem)
    meta["text"] = body.strip()
    meta["key_points"] = meta.get("key_points") or []
    return meta


def list_assignments() -> dict[str, dict]:
    items = {}
    for folder in (BUILTIN_ASSIGNMENTS, ASSIGN_DIR):
        for p in sorted(folder.glob("*.md")):
            a = _parse_md(p)
            items[a["id"]] = a
    return items


def _slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "-", text.lower(), flags=re.U).strip("-")
    return (s[:40] or "task") + "-" + uuid.uuid4().hex[:4]


def save_assignment(title: str, text: str, key_points: list[str],
                    instructions: str = "", hide_text: bool = True) -> str:
    aid = _slug(title)
    meta = {
        "id": aid, "title": title, "key_points": key_points,
        "instructions": instructions, "hide_text_during_recite": hide_text,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = ASSIGN_DIR / f"{aid}.md"
    path.write_text(
        "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n\n" + text.strip() + "\n",
        encoding="utf-8",
    )
    _push(path)
    return aid


# ---------- звіти ----------

def save_report(report: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    student = re.sub(r"[^\w-]+", "_", report.get("student", "anon"), flags=re.U)[:30]
    path = REPORT_DIR / f"{ts}_{report.get('assignment_id', 'x')}_{student}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _push(path)
    return path


def list_reports() -> list[dict]:
    out = []
    for p in sorted(REPORT_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out
