"""3R-бот: учень читає параграф, переказує його боту, бот уточнює і формує звіт для викладача."""

from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

# Streamlit Community Cloud віддає секрети через st.secrets — дублюємо їх у змінні середовища,
# щоб storage/llm читали конфігурацію однаково на будь-якому хостингу.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 — secrets.toml відсутній
    pass

import llm  # noqa: E402
import storage  # noqa: E402
from prompts import DIALOG_SYSTEM, REPORT_PROMPT  # noqa: E402

st.set_page_config(page_title="3R-бот", page_icon="📖", layout="centered")

TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "")
MAX_FOLLOWUPS = int(os.getenv("MAX_FOLLOWUPS", "4"))


@st.cache_resource
def _init_storage() -> bool:
    try:
        storage.pull_from_hub()
    except Exception as e:  # noqa: BLE001 — не валимо застосунок через недоступний Hub
        st.warning(f"Не вдалося завантажити дані з HF Dataset: {e}")
    return True


_init_storage()
ss = st.session_state


def system_prompt(a: dict) -> str:
    extra = f"\nДОДАТКОВІ ВКАЗІВКИ ВИКЛАДАЧА:\n{a['instructions']}\n" if a.get("instructions") else ""
    kp = "\n".join(f"- {p}" for p in a.get("key_points", [])) or "(не задано)"
    return DIALOG_SYSTEM.format(
        max_followups=MAX_FOLLOWUPS, extra_instructions=extra, title=a["title"],
        source_text=a["text"], key_points=kp,
    )


def report_markdown(r: dict) -> str:
    def bullets(xs): return "\n".join(f"- {x}" for x in xs) or "- —"
    return f"""# Звіт 3R: {r['assignment_title']}
**Учень:** {r['student']}  ·  **Клас/група:** {r.get('group', '')}  ·  **Дата:** {r['finished_at']}

| Повнота | Точність | Самостійність | Реплік учня |
|---|---|---|---|
| {r['completeness_percent']}% | {r['accuracy']}/5 | {r['independence']}/5 | {r['student_turns']} |

**Висновок:** {r['teacher_summary']}

**Розкрито:**
{bullets(r['covered_points'])}

**Пропущено:**
{bullets(r['missed_points'])}

**Помилки / хибні уявлення:**
{bullets(r['misconceptions'])}

**Сильні сторони:** {r['strengths']}

**Рекомендації:** {r['recommendations']}
"""


def transcript_text(history: list[dict]) -> str:
    return "\n\n".join(f"{'Учень' if m['role'] == 'user' else 'Бот'}: {m['content']}" for m in history)


# =====================================================================
# Режим учня
# =====================================================================

def student_view() -> None:
    assignments = storage.list_assignments()
    ss.setdefault("stage", "intro")

    if ss.stage == "intro":
        st.title("📖 Переказ параграфа (метод 3R)")
        st.markdown("**Read** — прочитай текст · **Recite** — перекажи своїми словами · **Review** — отримай відгук")

        preset = st.query_params.get("a")
        if preset and preset in assignments:
            aid = preset
            st.info(f"Завдання: **{assignments[aid]['title']}**")
        else:
            if not assignments:
                st.warning("Викладач ще не додав жодного завдання.")
                return
            aid = st.selectbox("Обери завдання", list(assignments), format_func=lambda k: assignments[k]["title"])

        name = st.text_input("Ім'я та прізвище")
        group = st.text_input("Клас / група")
        if st.button("Почати", type="primary", disabled=not name.strip()):
            ss.update(stage="read", aid=aid, student=name.strip(), group=group.strip(),
                      history=[], started_at=datetime.now().isoformat(timespec="seconds"))
            st.rerun()
        return

    a = assignments.get(ss.aid)
    if not a:
        st.error("Завдання не знайдено.")
        ss.stage = "intro"
        return

    if ss.stage == "read":
        st.subheader(f"1️⃣ Read: {a['title']}")
        st.caption("Уважно прочитай текст. Під час переказу його " +
                   ("не буде видно." if a.get("hide_text_during_recite", True) else "можна буде переглянути."))
        with st.container(border=True):
            st.markdown(a["text"])
        if st.button("Я прочитав(ла) — переходжу до переказу", type="primary"):
            ss.stage = "recite"
            st.rerun()
        return

    if ss.stage == "recite":
        st.subheader(f"2️⃣ Recite: {a['title']}")
        if not a.get("hide_text_during_recite", True):
            with st.expander("Переглянути текст"):
                st.markdown(a["text"])

        if not ss.history:
            st.chat_message("assistant").write(
                "Привіт! Перекажи, будь ласка, своїми словами, про що був цей параграф. "
                "Можна кількома повідомленнями.")
        for m in ss.history:
            st.chat_message("user" if m["role"] == "user" else "assistant").write(m["content"])

        if prompt := st.chat_input("Твій переказ…"):
            ss.history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                try:
                    reply = st.write_stream(llm.stream_reply(system_prompt(a), ss.history))
                except Exception as e:  # noqa: BLE001
                    ss.history.pop()
                    st.error(f"Помилка звернення до моделі: {e}")
                    return
            ss.history.append({"role": "assistant", "content": reply})

        student_turns = sum(m["role"] == "user" for m in ss.history)
        if st.button("✅ Завершити й отримати звіт", disabled=student_turns == 0):
            with st.spinner("Формую звіт…"):
                try:
                    rep = llm.make_report(system_prompt(a), ss.history, REPORT_PROMPT.format(
                        report_extra=a.get("report_instructions", "")))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Не вдалося сформувати звіт: {e}")
                    return
            report = {
                "assignment_id": a["id"], "assignment_title": a["title"],
                "student": ss.student, "group": ss.group,
                "started_at": ss.started_at, "finished_at": datetime.now().isoformat(timespec="seconds"),
                "student_turns": student_turns, **rep.model_dump(),
                "transcript": ss.history,
            }
            try:
                storage.save_report(report)
                report["_saved"] = True
            except Exception as e:  # noqa: BLE001
                report["_saved"] = False
                st.warning(f"Звіт не збережено на сервері ({e}). Завантаж його і надішли викладачу.")
            ss.report = report
            ss.stage = "done"
            st.rerun()
        return

    if ss.stage == "done":
        r = ss.report
        st.subheader("3️⃣ Review")
        c1, c2, c3 = st.columns(3)
        c1.metric("Повнота", f"{r['completeness_percent']}%")
        c2.metric("Точність", f"{r['accuracy']}/5")
        c3.metric("Самостійність", f"{r['independence']}/5")
        st.success(f"**Що вдалося:** {r['strengths']}")
        st.info(f"**Що повторити:** {r['recommendations']}")
        if r["missed_points"]:
            with st.expander("Що варто перечитати"):
                st.markdown("\n".join(f"- {p}" for p in r["missed_points"]))
        if r.get("_saved"):
            st.caption("Звіт автоматично надіслано викладачу.")
        md = report_markdown(r) + "\n---\n## Діалог\n\n" + transcript_text(r["transcript"])
        st.download_button("⬇️ Завантажити звіт", md, file_name=f"3R_{r['student']}.md", mime="text/markdown")
        if st.button("Нове завдання"):
            for k in ("stage", "history", "report", "aid"):
                ss.pop(k, None)
            st.rerun()


# =====================================================================
# Режим викладача
# =====================================================================

def read_upload(f) -> str:
    if f.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(f.read())).pages)
    return f.read().decode("utf-8", errors="ignore")


def teacher_view() -> None:
    st.title("👩‍🏫 Панель викладача")
    if not TEACHER_PASSWORD:
        st.error("Задайте TEACHER_PASSWORD у Secrets цього Space.")
        return
    if not ss.get("teacher_ok"):
        pw = st.text_input("Пароль", type="password")
        if st.button("Увійти"):
            ss.teacher_ok = pw == TEACHER_PASSWORD
            if not ss.teacher_ok:
                st.error("Невірний пароль")
            st.rerun()
        return

    st.caption(f"Сховище: {storage.hub_status()}")
    tab_reports, tab_new, tab_list = st.tabs(["📊 Звіти", "➕ Нове завдання", "📚 Завдання"])
    assignments = storage.list_assignments()

    with tab_reports:
        reports = storage.list_reports()
        if not reports:
            st.info("Звітів поки немає.")
        else:
            titles = {r["assignment_id"]: r["assignment_title"] for r in reports}
            pick = st.selectbox("Завдання", ["*"] + list(titles),
                                format_func=lambda k: "Усі" if k == "*" else titles[k])
            rows = [r for r in reports if pick == "*" or r["assignment_id"] == pick]
            df = pd.DataFrame([{
                "Дата": r["finished_at"], "Учень": r["student"], "Група": r.get("group", ""),
                "Завдання": r["assignment_title"], "Повнота %": r["completeness_percent"],
                "Точність": r["accuracy"], "Самостійність": r["independence"],
                "Висновок": r["teacher_summary"],
            } for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="3R_reports.csv", mime="text/csv")
            for r in rows:
                with st.expander(f"{r['student']} — {r['assignment_title']} — {r['completeness_percent']}%"):
                    st.markdown(report_markdown(r))
                    st.markdown("**Діалог:**")
                    st.text(transcript_text(r["transcript"]))

    with tab_new:
        with st.form("new_assignment", clear_on_submit=True):
            title = st.text_input("Назва (напр. «§12. Козацькі повстання 1620–1630-х»)")
            up = st.file_uploader("Текст параграфа з файлу (.txt, .md, .pdf)", type=["txt", "md", "pdf"])
            text = st.text_area("…або вставте текст параграфа", height=250)
            kp = st.text_area("Ключові ідеї, по одній на рядок (необов'язково — бот визначить сам)", height=120)
            instr = st.text_area("Додаткові вказівки боту (необов'язково)",
                                 placeholder="Напр.: звертай увагу на дати та імена; вік учнів — 8 клас.")
            hide = st.checkbox("Ховати текст під час переказу", value=True)
            if st.form_submit_button("Зберегти", type="primary"):
                body = read_upload(up) if up else text
                if not title.strip() or not body.strip():
                    st.error("Потрібні назва і текст.")
                else:
                    aid = storage.save_assignment(
                        title.strip(), body, [l.strip("-• ").strip() for l in kp.splitlines() if l.strip()],
                        instr.strip(), hide)
                    st.success(f"Збережено. Посилання для учнів: `?a={aid}` (додайте до адреси застосунку)")

    with tab_list:
        for aid, a in assignments.items():
            with st.expander(a["title"]):
                st.code(f"?a={aid}", language=None)
                st.caption("Додайте це до адреси застосунку, щоб учні одразу потрапили на завдання.")
                st.markdown(a["text"][:1500] + ("…" if len(a["text"]) > 1500 else ""))


mode = st.sidebar.radio("Режим", ["Учень", "Викладач"], label_visibility="collapsed")
st.sidebar.caption(f"Модель: `{llm.MODEL}`")
student_view() if mode == "Учень" else teacher_view()
