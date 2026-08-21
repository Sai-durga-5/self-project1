from datetime import datetime
from agents.state import InterviewState
from database.override_log import insert_log


def log_agent(state: InterviewState) -> InterviewState:
    """
    Log agent: Records all AI decisions and human overrides to SQLite.
    Called after both human checkpoints to create a full audit trail.
    """
    timestamp = datetime.now().isoformat()
    override_log = list(state.get("override_log", []))

    scored_candidates = state.get("scored_candidates", [])
    shortlist = state.get("shortlist", [])
    final_decisions = state.get("final_decisions", {})
    human_decision_1 = state.get("human_decision_1", "")
    human_decision_2 = state.get("human_decision_2", "")

    shortlist_names = {c["name"] for c in shortlist}

    # Log checkpoint 1 results (screening decisions)
    for candidate in scored_candidates:
        name = candidate["name"]
        ai_score = candidate.get("score", 0)
        ai_recommendation = candidate.get("recommendation", "")

        # Determine if human edited the score
        human_edited_score = None
        for entry in override_log:
            if entry.get("candidate_name") == name and entry.get("step") == "checkpoint_1":
                human_edited_score = entry.get("human_edited_score")
                break

        # Was this candidate shortlisted?
        shortlisted = name in shortlist_names
        final_outcome = "SHORTLISTED" if shortlisted else "REJECTED_AT_SCREENING"

        record = {
            "timestamp": timestamp,
            "step": "screening",
            "candidate_name": name,
            "ai_score": ai_score,
            "ai_recommendation": ai_recommendation,
            "human_decision": human_decision_1,
            "human_edited_score": human_edited_score,
            "final_outcome": final_outcome,
            "notes": f"Rank: {candidate.get('rank', 'N/A')}",
        }
        insert_log(record)

    # Log checkpoint 2 results (final hire/reject decisions)
    for candidate in shortlist:
        name = candidate["name"]
        final_decision = final_decisions.get(name, "PENDING")
        ai_score = candidate.get("score", 0)
        ai_recommendation = candidate.get("recommendation", "")

        record = {
            "timestamp": timestamp,
            "step": "final_decision",
            "candidate_name": name,
            "ai_score": ai_score,
            "ai_recommendation": ai_recommendation,
            "human_decision": human_decision_2,
            "human_edited_score": None,
            "final_outcome": final_decision,
            "notes": f"Questions reviewed: {len(state.get('interview_questions', {}).get(name, []))}",
        }
        insert_log(record)

    return {
        **state,
        "current_step": "done",
    }
