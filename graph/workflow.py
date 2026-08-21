"""
LangGraph workflow definition for the Agentic AI Interview Manager.

Graph flow:
START
  → screening_agent
  → human_checkpoint_1  [interrupt_before]
  → [conditional]: APPROVED/EDITED → question_agent | REJECTED → screening_agent
  → human_checkpoint_2  [interrupt_before]
  → log_agent
  → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import InterviewState
from agents.screening_agent import screening_agent
from agents.question_agent import question_agent
from agents.log_agent import log_agent
from human_checkpoints.checkpoints import checkpoint_1, checkpoint_2, route_after_checkpoint_1


def build_workflow() -> StateGraph:
    """Build and return the compiled LangGraph workflow."""
    graph = StateGraph(InterviewState)

    # Add nodes
    graph.add_node("screening_agent", screening_agent)
    graph.add_node("human_checkpoint_1", checkpoint_1)
    graph.add_node("question_agent", question_agent)
    graph.add_node("human_checkpoint_2", checkpoint_2)
    graph.add_node("log_agent", log_agent)

    # Add edges
    graph.add_edge(START, "screening_agent")
    graph.add_edge("screening_agent", "human_checkpoint_1")

    # Conditional edge after checkpoint_1
    graph.add_conditional_edges(
        "human_checkpoint_1",
        route_after_checkpoint_1,
        {
            "question_agent": "question_agent",
            "screening_agent": "screening_agent",
        },
    )

    graph.add_edge("question_agent", "human_checkpoint_2")
    graph.add_edge("human_checkpoint_2", "log_agent")
    graph.add_edge("log_agent", END)

    # Compile with memory checkpointer and interrupt before human checkpoints
    memory = MemorySaver()
    compiled = graph.compile(
        checkpointer=memory,
        interrupt_before=["human_checkpoint_1", "human_checkpoint_2"],
    )
    return compiled


# Module-level compiled app (lazy-initialized)
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_workflow()
    return _app
