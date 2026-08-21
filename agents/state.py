from typing import TypedDict, Optional


class InterviewState(TypedDict):
    """
    Central state object passed through the entire LangGraph pipeline.
    """
    # Input
    job_description: str
    resumes: list[dict]               # [{"name": str, "text": str, "filename": str}]

    # After screening agent
    scored_candidates: list[dict]     # [{"name": str, "score": int, "reasoning": str,
                                      #   "strengths": list, "weaknesses": list,
                                      #   "recommendation": str, "rank": int}]

    # After human checkpoint 1
    shortlist: list[dict]             # subset of scored_candidates approved by recruiter

    # After question agent
    interview_questions: dict         # {candidate_name: [{"question": str, "type": str,
                                      #                    "expected_answer_points": list}]}

    # After human checkpoint 2
    final_decisions: dict             # {candidate_name: "HIRE" | "REJECT"}

    # Human decisions
    human_decision_1: str             # "APPROVED" | "EDITED" | "REJECTED"
    human_decision_2: str             # "APPROVED" | "EDITED"

    # Audit trail
    override_log: list[dict]

    # Pipeline tracking
    current_step: str                 # "setup" | "screening" | "checkpoint_1" |
                                      # "questions" | "checkpoint_2" | "logging" | "done"

    # RAG context (for UI display)
    rag_context_screening: list[str]
    rag_context_questions: dict       # {candidate_name: list[str]}
