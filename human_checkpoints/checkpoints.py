"""
Human checkpoint handlers.

These functions act as pass-through nodes in the LangGraph workflow.
The actual human interaction is handled by the Streamlit UI via session_state flags.
These functions simply mark the state so the UI knows to pause and await input.
"""

from agents.state import InterviewState


def checkpoint_1(state: InterviewState) -> InterviewState:
    """
    Human Checkpoint 1: After screening agent.
    The Streamlit UI pauses here to let the recruiter review/edit/approve the shortlist.
    This node sets the current_step so the UI knows to show the approval interface.
    """
    return {
        **state,
        "current_step": "checkpoint_1",
    }


def checkpoint_2(state: InterviewState) -> InterviewState:
    """
    Human Checkpoint 2: After question generation agent.
    The Streamlit UI pauses here to let the recruiter review questions and make hire/reject decisions.
    """
    return {
        **state,
        "current_step": "checkpoint_2",
    }


def route_after_checkpoint_1(state: InterviewState) -> str:
    """
    Conditional edge after checkpoint_1.
    - APPROVED or EDITED → proceed to question_agent
    - REJECTED → loop back to screening_agent
    """
    decision = state.get("human_decision_1", "")
    if decision in ("APPROVED", "EDITED"):
        return "question_agent"
    elif decision == "REJECTED":
        return "screening_agent"
    # Default to question agent if somehow unset
    return "question_agent"
