"""
Streamlit UI for the Agentic AI Interview Manager.
"""

import os
import sys
import io
import re
import csv
import json
from datetime import datetime
import torch
torch.classes.__path__ = []  # Fix Streamlit watcher conflict
# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

from database.override_log import init_db, get_all_logs, get_logs_by_candidate
from agents.state import InterviewState
from graph.workflow import get_app

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
DEFAULTS = {
    "page": "Setup",
    "pipeline_status": "idle",          # idle | running | waiting_checkpoint_1 | waiting_checkpoint_2 | done | error
    "interview_state": None,
    "thread_id": "interview_thread_1",
    "error_message": None,
    "langgraph_app": None,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _get_app():
    if st.session_state.langgraph_app is None:
        st.session_state.langgraph_app = get_app()
    return st.session_state.langgraph_app


# ─────────────────────────────────────────────
# PDF parsing
# ─────────────────────────────────────────────
def parse_pdf(uploaded_file) -> str:
    """Extract and clean text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        raw = "\n".join(text_parts)
        # Clean: collapse multiple whitespace, remove non-printable chars
        raw = re.sub(r"[^\x20-\x7E\n]", " ", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()
    except Exception as e:
        return f"[Error parsing PDF: {e}]"


def candidate_name_from_filename(filename: str) -> str:
    """Derive a display name from the filename (strip extension, replace underscores)."""
    base = os.path.splitext(filename)[0]
    return base.replace("_", " ").replace("-", " ").strip()


# ─────────────────────────────────────────────
# Recommendation badge helpers
# ─────────────────────────────────────────────
RECOMMENDATION_COLORS = {
    "STRONG_YES": "#28a745",
    "YES": "#85c285",
    "MAYBE": "#ffc107",
    "NO": "#dc3545",
}

RECOMMENDATION_LABELS = {
    "STRONG_YES": "Strong Yes",
    "YES": "Yes",
    "MAYBE": "Maybe",
    "NO": "No",
}

QUESTION_TYPE_COLORS = {
    "Technical": "#0d6efd",
    "Behavioral": "#6f42c1",
    "Situational": "#fd7e14",
    "Culture Fit": "#20c997",
    "General": "#6c757d",
}


def recommendation_badge(rec: str) -> str:
    color = RECOMMENDATION_COLORS.get(rec, "#6c757d")
    label = RECOMMENDATION_LABELS.get(rec, rec)
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:bold">{label}</span>'


def question_type_badge(qtype: str) -> str:
    color = QUESTION_TYPE_COLORS.get(qtype, "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem">{qtype}</span>'


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("AI Interview Manager")
        st.markdown("---")

        pages = ["Setup", "Screening Results", "Interview Questions", "Audit Log"]
        icons = ["⚙️", "📋", "❓", "📊"]

        for icon, page in zip(icons, pages):
            # Highlight active
            is_active = st.session_state.page == page
            label = f"**{icon} {page}**" if is_active else f"{icon} {page}"
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")
        st.markdown("**Pipeline Status**")
        status = st.session_state.pipeline_status
        status_map = {
            "idle": ("⚪", "Ready"),
            "running": ("🔄", "Running..."),
            "waiting_checkpoint_1": ("⏸️", "Awaiting Checkpoint 1"),
            "waiting_checkpoint_2": ("⏸️", "Awaiting Checkpoint 2"),
            "done": ("✅", "Complete"),
            "error": ("❌", "Error"),
        }
        icon_s, label_s = status_map.get(status, ("⚪", status))
        st.markdown(f"{icon_s} {label_s}")

        if st.session_state.error_message:
            st.error(st.session_state.error_message)


# ─────────────────────────────────────────────
# PAGE 1 — Setup
# ─────────────────────────────────────────────
def page_setup():
    st.title("Setup — Upload Resumes & Job Description")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the full job description here:",
            height=300,
            value=st.session_state.get("saved_jd", ""),
            placeholder="e.g., We are looking for a Senior Software Engineer with 5+ years experience in Python and distributed systems...",
            key="jd_input",
        )

    with col2:
        st.subheader("Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF resumes (multiple allowed):",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_uploader",
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} resume(s) uploaded")
            for f in uploaded_files:
                st.markdown(f"- 📄 {f.name}")

    st.markdown("---")

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        start_clicked = st.button(
            "Start Interview Pipeline",
            type="primary",
            disabled=st.session_state.pipeline_status == "running",
        )

    with col_status:
        status = st.session_state.pipeline_status
        if status == "idle":
            st.info("Pipeline ready. Upload resumes and job description, then click Start.")
        elif status == "running":
            st.warning("Pipeline is running... please wait.")
        elif status in ("waiting_checkpoint_1", "waiting_checkpoint_2"):
            st.warning("Pipeline paused — human input required. See other pages.")
        elif status == "done":
            st.success("Pipeline complete! View results in Screening Results and Audit Log.")
        elif status == "error":
            st.error(f"Error: {st.session_state.error_message}")

    if start_clicked:
        if not job_description.strip():
            st.error("Please enter a job description before starting.")
            return
        if not uploaded_files:
            st.error("Please upload at least one resume PDF.")
            return

        # Parse PDFs
        resumes = []
        with st.spinner("Parsing PDFs..."):
            for f in uploaded_files:
                text = parse_pdf(f)
                name = candidate_name_from_filename(f.name)
                resumes.append({
                    "name": name,
                    "text": text,
                    "filename": f.name,
                })

        # Build initial state
        initial_state: InterviewState = {
            "job_description": job_description,
            "resumes": resumes,
            "scored_candidates": [],
            "shortlist": [],
            "interview_questions": {},
            "final_decisions": {},
            "human_decision_1": "",
            "human_decision_2": "",
            "override_log": [],
            "current_step": "setup",
            "rag_context_screening": [],
            "rag_context_questions": {},
        }

        st.session_state.saved_jd = job_description
        st.session_state.pipeline_status = "running"
        st.session_state.error_message = None

        # Run the graph until first interrupt
        try:
            app = _get_app()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            with st.spinner("Running screening agent (calling GPT-4o)..."):
                # Invoke until interrupt
                result = app.invoke(initial_state, config=config)

            # After invoke, graph pauses before human_checkpoint_1
            # Get the current state snapshot
            snapshot = app.get_state(config)
            current_values = snapshot.values

            st.session_state.interview_state = dict(current_values)
            st.session_state.pipeline_status = "waiting_checkpoint_1"
            st.session_state.page = "Screening Results"
            st.rerun()

        except Exception as e:
            st.session_state.pipeline_status = "error"
            st.session_state.error_message = str(e)
            st.error(f"Pipeline error: {e}")


# ─────────────────────────────────────────────
# PAGE 2 — Screening Results (Checkpoint 1)
# ─────────────────────────────────────────────
def page_screening():
    st.title("Screening Results — Human Checkpoint 1")

    state = st.session_state.interview_state
    status = st.session_state.pipeline_status

    if state is None or not state.get("scored_candidates"):
        st.info("No screening results yet. Go to Setup to start the pipeline.")
        return

    scored = state["scored_candidates"]
    rag_context = state.get("rag_context_screening", [])

    if status == "waiting_checkpoint_1":
        st.warning("Pipeline paused — review candidates below and approve/edit the shortlist.")
    elif status in ("waiting_checkpoint_2", "done"):
        st.success("Checkpoint 1 completed.")
    elif status == "running":
        st.info("Screening in progress...")
        return

    # ── Summary table ──
    st.subheader(f"Candidates ({len(scored)} screened)")
    summary_data = []
    for c in scored:
        summary_data.append({
            "Rank": c.get("rank", "—"),
            "Name": c["name"],
            "Score": c["score"],
            "Recommendation": c["recommendation"],
        })
    summary_df = pd.DataFrame(summary_data)

    def color_row(row):
        rec = row["Recommendation"]
        if rec == "STRONG_YES":
            return ["background-color: #d4edda"] * len(row)
        elif rec == "YES":
            return ["background-color: #e8f5e9"] * len(row)
        elif rec == "MAYBE":
            return ["background-color: #fff3cd"] * len(row)
        else:
            return ["background-color: #f8d7da"] * len(row)

    st.dataframe(
        summary_df.style.apply(color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Per-candidate review ──
    st.subheader("Review & Edit Candidates")

    # Initialize per-candidate session state
    if "cp1_include" not in st.session_state:
        st.session_state.cp1_include = {
            c["name"]: c["recommendation"] in ("STRONG_YES", "YES", "MAYBE")
            for c in scored
        }
    if "cp1_scores" not in st.session_state:
        st.session_state.cp1_scores = {c["name"]: c["score"] for c in scored}
    if "cp1_notes" not in st.session_state:
        st.session_state.cp1_notes = {c["name"]: "" for c in scored}

    for c in scored:
        name = c["name"]
        rec = c["recommendation"]
        badge_html = recommendation_badge(rec)

        with st.expander(f"{'✅' if st.session_state.cp1_include.get(name, False) else '❌'} {name}  |  Score: {c['score']}  |  Rec: {rec}", expanded=False):
            col_left, col_right = st.columns([3, 1])

            with col_left:
                st.markdown(f"**Reasoning:** {c.get('reasoning', 'N/A')}")

                if c.get("strengths"):
                    st.markdown("**Strengths:**")
                    for s in c["strengths"]:
                        st.markdown(f"  - ✅ {s}")

                if c.get("weaknesses"):
                    st.markdown("**Areas for Improvement:**")
                    for w in c["weaknesses"]:
                        st.markdown(f"  - ⚠️ {w}")

            with col_right:
                st.markdown(badge_html, unsafe_allow_html=True)
                include = st.checkbox(
                    "Include in shortlist",
                    value=st.session_state.cp1_include.get(name, False),
                    key=f"include_{name}",
                )
                st.session_state.cp1_include[name] = include

                new_score = st.number_input(
                    "Edit score (0–100)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.cp1_scores.get(name, c["score"]),
                    step=1,
                    key=f"score_{name}",
                )
                st.session_state.cp1_scores[name] = new_score

                notes = st.text_area(
                    "Notes",
                    value=st.session_state.cp1_notes.get(name, ""),
                    height=80,
                    key=f"notes_{name}",
                )
                st.session_state.cp1_notes[name] = notes

    # ── RAG context ──
    if rag_context:
        with st.expander("RAG Context Used (similar job descriptions retrieved)", expanded=False):
            for i, ctx in enumerate(rag_context):
                st.markdown(f"**Context {i+1}:**")
                st.text(ctx[:500])
                st.markdown("---")

    st.markdown("---")

    # ── Checkpoint actions ──
    if status == "waiting_checkpoint_1":
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Approve Shortlist", type="primary", use_container_width=True):
                _submit_checkpoint_1(scored, decision="APPROVED")

        with col2:
            if st.button("Reject All & Re-screen", type="secondary", use_container_width=True):
                if st.session_state.get("confirm_reject"):
                    _submit_checkpoint_1(scored, decision="REJECTED")
                    st.session_state.confirm_reject = False
                else:
                    st.session_state.confirm_reject = True
                    st.warning("Click 'Reject All & Re-screen' again to confirm restarting the screening.")

    elif status in ("waiting_checkpoint_2", "done"):
        st.info("Checkpoint 1 already submitted. Shortlist is locked.")

        shortlist = state.get("shortlist", [])
        if shortlist:
            st.subheader("Approved Shortlist")
            for c in shortlist:
                st.markdown(f"- ✅ **{c['name']}** — Score: {c.get('score', 'N/A')} | {c.get('recommendation', '')}")


def _submit_checkpoint_1(scored: list[dict], decision: str):
    """Apply human edits and continue the LangGraph pipeline past checkpoint 1."""
    app = _get_app()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = st.session_state.interview_state

    if decision == "REJECTED":
        # Restart: clear scored candidates and re-run screening
        updated_state = {
            **state,
            "human_decision_1": "REJECTED",
            "scored_candidates": [],
            "shortlist": [],
            "current_step": "screening",
        }
        # Update the graph state
        app.update_state(config, updated_state, as_node="human_checkpoint_1")
        st.session_state.pipeline_status = "running"

        try:
            with st.spinner("Re-running screening agent..."):
                app.invoke(None, config=config)
            snapshot = app.get_state(config)
            st.session_state.interview_state = dict(snapshot.values)
            st.session_state.pipeline_status = "waiting_checkpoint_1"
            st.rerun()
        except Exception as e:
            st.session_state.pipeline_status = "error"
            st.session_state.error_message = str(e)
            st.error(f"Error: {e}")
        return

    # Build shortlist from human selections
    include_map = st.session_state.get("cp1_include", {})
    score_map = st.session_state.get("cp1_scores", {})
    notes_map = st.session_state.get("cp1_notes", {})

    # Detect if any edits were made
    any_edited = False
    shortlist = []
    override_entries = list(state.get("override_log", []))

    for c in scored:
        name = c["name"]
        original_score = c["score"]
        new_score = score_map.get(name, original_score)
        included = include_map.get(name, False)

        if included:
            updated_candidate = dict(c)
            updated_candidate["score"] = new_score
            shortlist.append(updated_candidate)

        if new_score != original_score:
            any_edited = True
            override_entries.append({
                "candidate_name": name,
                "step": "checkpoint_1",
                "ai_score": original_score,
                "human_edited_score": new_score,
                "notes": notes_map.get(name, ""),
                "timestamp": datetime.now().isoformat(),
            })

    final_decision = "EDITED" if any_edited else "APPROVED"
    if not shortlist:
        st.error("No candidates selected for shortlist. Please include at least one candidate.")
        return

    updated_state = {
        **state,
        "shortlist": shortlist,
        "human_decision_1": final_decision,
        "override_log": override_entries,
        "current_step": "questions",
    }

    # Update LangGraph state and continue
    app.update_state(config, updated_state, as_node="human_checkpoint_1")
    st.session_state.pipeline_status = "running"

    try:
        with st.spinner(f"Generating interview questions for {len(shortlist)} candidates..."):
            app.invoke(None, config=config)

        snapshot = app.get_state(config)
        st.session_state.interview_state = dict(snapshot.values)
        st.session_state.pipeline_status = "waiting_checkpoint_2"
        st.session_state.page = "Interview Questions"
        st.rerun()
    except Exception as e:
        st.session_state.pipeline_status = "error"
        st.session_state.error_message = str(e)
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# PAGE 3 — Interview Questions (Checkpoint 2)
# ─────────────────────────────────────────────
def page_questions():
    st.title("Interview Questions — Human Checkpoint 2")

    state = st.session_state.interview_state
    status = st.session_state.pipeline_status

    if state is None or not state.get("interview_questions"):
        if state and state.get("shortlist") and status == "running":
            st.info("Interview questions are being generated... please wait.")
        else:
            st.info("No interview questions yet. Complete Checkpoint 1 first.")
        return

    shortlist = state.get("shortlist", [])
    questions_map = state.get("interview_questions", {})
    rag_context_map = state.get("rag_context_questions", {})

    if status == "waiting_checkpoint_2":
        st.warning("Pipeline paused — review and edit questions, then finalize hire/reject decisions.")
    elif status == "done":
        st.success("Checkpoint 2 completed. Final decisions locked.")

    # Initialize session state for questions and decisions
    if "cp2_questions" not in st.session_state:
        st.session_state.cp2_questions = {}
        for cname, qs in questions_map.items():
            st.session_state.cp2_questions[cname] = [dict(q) for q in qs]

    if "cp2_decisions" not in st.session_state:
        st.session_state.cp2_decisions = {c["name"]: "HIRE" for c in shortlist}

    for candidate in shortlist:
        name = candidate["name"]
        score = candidate.get("score", "N/A")
        rec = candidate.get("recommendation", "")
        qs = st.session_state.cp2_questions.get(name, questions_map.get(name, []))

        st.markdown(f"---")
        col_title, col_badge = st.columns([4, 1])
        with col_title:
            st.subheader(f"{name}")
            st.caption(f"Score: {score} | AI Recommendation: {rec}")
        with col_badge:
            st.markdown(recommendation_badge(rec), unsafe_allow_html=True)

        # HIRE / REJECT toggle
        hire_col, reject_col = st.columns(2)
        current_decision = st.session_state.cp2_decisions.get(name, "HIRE")
        with hire_col:
            if st.button(
                f"{'✅ ' if current_decision == 'HIRE' else ''}HIRE",
                key=f"hire_{name}",
                type="primary" if current_decision == "HIRE" else "secondary",
                use_container_width=True,
            ):
                st.session_state.cp2_decisions[name] = "HIRE"
                st.rerun()
        with reject_col:
            if st.button(
                f"{'❌ ' if current_decision == 'REJECT' else ''}REJECT",
                key=f"reject_{name}",
                type="primary" if current_decision == "REJECT" else "secondary",
                use_container_width=True,
            ):
                st.session_state.cp2_decisions[name] = "REJECT"
                st.rerun()

        st.markdown(f"**Decision: {'✅ HIRE' if current_decision == 'HIRE' else '❌ REJECT'}**")
        st.markdown("")

        # Questions
        st.markdown("**Interview Questions:**")
        for i, q in enumerate(qs):
            q_type = q.get("type", "General")
            badge = question_type_badge(q_type)

            with st.expander(f"Q{i+1}: {q.get('question', '')[:80]}...", expanded=(i < 2)):
                st.markdown(f"Type: {badge}", unsafe_allow_html=True)

                edited_question = st.text_area(
                    "Question",
                    value=q.get("question", ""),
                    height=80,
                    key=f"q_{name}_{i}_text",
                )
                if edited_question != q.get("question", ""):
                    st.session_state.cp2_questions[name][i]["question"] = edited_question

                expected_pts = q.get("expected_answer_points", [])
                if expected_pts:
                    st.markdown("**Expected Answer Points:**")
                for pt in expected_pts:
                    st.markdown(f"  - {pt}")

        # RAG context per candidate
        rag_ctx = rag_context_map.get(name, [])
        if rag_ctx:
            with st.expander(f"RAG Context Used for {name}", expanded=False):
                for i, ctx in enumerate(rag_ctx):
                    st.markdown(f"**Context {i+1}:**")
                    st.text(ctx[:400])
                    st.markdown("---")

    st.markdown("---")

    if status == "waiting_checkpoint_2":
        if st.button("Finalize & Log Decisions", type="primary", use_container_width=True):
            _submit_checkpoint_2(shortlist)

    elif status == "done":
        st.info("Decisions finalized and logged. Check Audit Log for details.")
        final_decisions = state.get("final_decisions", {})
        if final_decisions:
            st.subheader("Final Decisions")
            for name, decision in final_decisions.items():
                icon = "✅" if decision == "HIRE" else "❌"
                st.markdown(f"- {icon} **{name}**: {decision}")


def _submit_checkpoint_2(shortlist: list[dict]):
    """Apply final decisions and complete the pipeline."""
    app = _get_app()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = st.session_state.interview_state

    final_decisions = dict(st.session_state.get("cp2_decisions", {}))
    edited_questions = dict(st.session_state.get("cp2_questions", {}))

    # Detect edits
    original_questions = state.get("interview_questions", {})
    any_edited = edited_questions != original_questions

    updated_state = {
        **state,
        "interview_questions": edited_questions,
        "final_decisions": final_decisions,
        "human_decision_2": "EDITED" if any_edited else "APPROVED",
        "current_step": "logging",
    }

    app.update_state(config, updated_state, as_node="human_checkpoint_2")
    st.session_state.pipeline_status = "running"

    try:
        with st.spinner("Logging decisions to database..."):
            app.invoke(None, config=config)

        snapshot = app.get_state(config)
        st.session_state.interview_state = dict(snapshot.values)
        st.session_state.pipeline_status = "done"
        st.session_state.page = "Audit Log"
        st.rerun()
    except Exception as e:
        st.session_state.pipeline_status = "error"
        st.session_state.error_message = str(e)
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# PAGE 4 — Audit Log
# ─────────────────────────────────────────────
def page_audit_log():
    st.title("Audit Log — AI vs Human Decisions")

    logs = get_all_logs()

    if not logs:
        st.info("No log entries yet. Complete the full pipeline to see results here.")
        return

    df = pd.DataFrame(logs)

    # ── Stats ──
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    total_screened = df[df["step"] == "screening"]["candidate_name"].nunique()
    shortlisted = df[(df["step"] == "screening") & (df["final_outcome"] == "SHORTLISTED")]["candidate_name"].nunique()
    hired = df[(df["step"] == "final_decision") & (df["final_outcome"] == "HIRE")].shape[0]
    overrides = df[df["human_edited_score"].notna()].shape[0]

    with col1:
        st.metric("Total Screened", total_screened)
    with col2:
        st.metric("Shortlisted", shortlisted)
    with col3:
        st.metric("Hired", hired)
    with col4:
        st.metric("Score Overrides", overrides)

    st.markdown("---")

    # ── Filter ──
    st.subheader("Filter Logs")
    candidate_names = sorted(df["candidate_name"].dropna().unique().tolist())
    selected_candidate = st.selectbox(
        "Filter by candidate:",
        options=["All"] + candidate_names,
    )

    if selected_candidate != "All":
        df_filtered = df[df["candidate_name"] == selected_candidate]
    else:
        df_filtered = df

    # ── Table ──
    st.subheader(f"Log Entries ({len(df_filtered)} rows)")
    display_cols = [
        "timestamp", "step", "candidate_name", "ai_score",
        "ai_recommendation", "human_decision", "human_edited_score",
        "final_outcome", "notes"
    ]
    existing_cols = [c for c in display_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[existing_cols].rename(columns={
            "timestamp": "Timestamp",
            "step": "Step",
            "candidate_name": "Candidate",
            "ai_score": "AI Score",
            "ai_recommendation": "AI Rec.",
            "human_decision": "Human Decision",
            "human_edited_score": "Edited Score",
            "final_outcome": "Final Outcome",
            "notes": "Notes",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Download CSV ──
    csv_buffer = io.StringIO()
    df_filtered[existing_cols].to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"interview_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────
def main():
    render_sidebar()

    page = st.session_state.page

    if page == "Setup":
        page_setup()
    elif page == "Screening Results":
        page_screening()
    elif page == "Interview Questions":
        page_questions()
    elif page == "Audit Log":
        page_audit_log()


if __name__ == "__main__":
    main()
