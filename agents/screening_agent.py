import os
import json
import re
from dotenv import load_dotenv
from agents.state import InterviewState
from rag.vectorstore import query as rag_query

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

SYSTEM_PROMPT = """You are an expert HR recruiter with 10 years experience. You will be given a job description and a candidate resume. Score the candidate from 0-100 and provide detailed reasoning. Consider: skills match, experience level, education, career progression. Return JSON with keys: score (int), reasoning (str), strengths (list), weaknesses (list), recommendation (str: STRONG_YES/YES/MAYBE/NO)"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    return json.loads(text)


def _score_candidate(
    job_description: str,
    resume_text: str,
    rag_context: list,
    candidate_name: str,
    retry: bool = False,
) -> dict:
    context_block = "\n\n".join(rag_context[:3]) if rag_context else ""

    user_message = f"""JOB DESCRIPTION:
{job_description}

SIMILAR ROLE BENCHMARKS FROM DATABASE:
{context_block}

CANDIDATE RESUME ({candidate_name}):
{resume_text[:3000]}

Score this candidate and return ONLY valid JSON."""

    if retry:
        user_message += "\n\nIMPORTANT: Return ONLY valid JSON, no other text, no markdown."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content
    return _extract_json(raw)


def screening_agent(state: InterviewState) -> InterviewState:
    """
    Agent 1: Score and rank all uploaded resumes against the job description.
    Uses RAG to retrieve similar job profiles as benchmarks.
    """
    job_description = state["job_description"]
    resumes = state["resumes"]

    # RAG: retrieve similar job descriptions as context/benchmarks
    rag_context = rag_query("job_descriptions", job_description, n_results=5)

    scored = []
    for resume in resumes:
        candidate_name = resume["name"]
        resume_text = resume["text"]

        try:
            result = _score_candidate(
                job_description=job_description,
                resume_text=resume_text,
                rag_context=rag_context,
                candidate_name=candidate_name,
            )
        except (json.JSONDecodeError, Exception):
            try:
                result = _score_candidate(
                    job_description=job_description,
                    resume_text=resume_text,
                    rag_context=rag_context,
                    candidate_name=candidate_name,
                    retry=True,
                )
            except Exception as e:
                result = {
                    "score": 0,
                    "reasoning": f"Error during scoring: {str(e)}",
                    "strengths": [],
                    "weaknesses": ["Could not parse resume"],
                    "recommendation": "NO",
                }

        scored.append({
            "name": candidate_name,
            "filename": resume.get("filename", candidate_name),
            "score": int(result.get("score", 0)),
            "reasoning": result.get("reasoning", ""),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "recommendation": result.get("recommendation", "NO"),
            "rank": 0,
        })

    # Sort by score descending and assign ranks
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, candidate in enumerate(scored):
        candidate["rank"] = i + 1

    return {
        **state,
        "scored_candidates": scored,
        "rag_context_screening": rag_context,
        "current_step": "checkpoint_1",
    }