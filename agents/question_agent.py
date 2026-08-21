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

SYSTEM_PROMPT = """You are an expert technical interviewer. Given the job description and this candidate's resume, generate 8 tailored interview questions. Mix: 3 technical questions specific to their skills, 3 behavioral questions based on their experience gaps, 2 situational/culture fit questions. Return JSON with key: questions (list of dicts with 'question', 'type', 'expected_answer_points')"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    return json.loads(text)


def _find_resume_text(state: InterviewState, candidate_name: str) -> str:
    """Find the resume text for a given candidate."""
    for resume in state["resumes"]:
        if resume["name"] == candidate_name:
            return resume["text"]
    return ""


def _generate_questions(
    job_description: str,
    resume_text: str,
    rag_context: list,
    candidate_name: str,
    candidate_info: dict,
    retry: bool = False,
) -> dict:
    context_block = "\n\n".join(rag_context[:3]) if rag_context else ""

    strengths = ", ".join(candidate_info.get("strengths", []))
    weaknesses = ", ".join(candidate_info.get("weaknesses", []))

    user_message = f"""JOB DESCRIPTION:
{job_description}

CANDIDATE: {candidate_name}
Score: {candidate_info.get('score', 'N/A')} | Recommendation: {candidate_info.get('recommendation', 'N/A')}
Strengths: {strengths}
Areas for improvement: {weaknesses}

RELEVANT INTERVIEW QUESTIONS FROM DATABASE:
{context_block}

CANDIDATE RESUME (excerpt):
{resume_text[:2500]}

Generate 8 tailored interview questions and return ONLY valid JSON."""

    if retry:
        user_message += "\n\nIMPORTANT: Return ONLY valid JSON, no other text, no markdown."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content
    return _extract_json(raw)


def question_agent(state: InterviewState) -> InterviewState:
    """
    Agent 2: Generate 8 tailored interview questions for each shortlisted candidate.
    Uses RAG to retrieve relevant interview Q&A pairs.
    """
    job_description = state["job_description"]
    shortlist = state["shortlist"]

    interview_questions = dict(state.get("interview_questions", {}))
    rag_context_questions = dict(state.get("rag_context_questions", {}))

    scored_map = {c["name"]: c for c in state.get("scored_candidates", [])}

    for candidate in shortlist:
        candidate_name = candidate["name"]
        resume_text = _find_resume_text(state, candidate_name)
        candidate_info = scored_map.get(candidate_name, candidate)

        strengths_text = " ".join(candidate_info.get("strengths", []))
        rag_query_text = f"{job_description[:500]} {strengths_text}"
        rag_context = rag_query("interview_questions", rag_query_text, n_results=5)
        rag_context_questions[candidate_name] = rag_context

        try:
            result = _generate_questions(
                job_description=job_description,
                resume_text=resume_text,
                rag_context=rag_context,
                candidate_name=candidate_name,
                candidate_info=candidate_info,
            )
        except (json.JSONDecodeError, Exception):
            try:
                result = _generate_questions(
                    job_description=job_description,
                    resume_text=resume_text,
                    rag_context=rag_context,
                    candidate_name=candidate_name,
                    candidate_info=candidate_info,
                    retry=True,
                )
            except Exception as e:
                result = {
                    "questions": [
                        {
                            "question": f"Could you walk us through your background? (Error generating questions: {str(e)})",
                            "type": "Behavioral",
                            "expected_answer_points": ["General background", "Relevant experience"],
                        }
                    ]
                }

        questions = result.get("questions", [])

        normalized = []
        for q in questions[:8]:
            normalized.append({
                "question": q.get("question", ""),
                "type": q.get("type", "General"),
                "expected_answer_points": q.get("expected_answer_points", []),
            })

        while len(normalized) < 8:
            normalized.append({
                "question": "Describe a challenging situation you faced and how you resolved it.",
                "type": "Behavioral",
                "expected_answer_points": ["Problem identification", "Actions taken", "Outcome"],
            })

        interview_questions[candidate_name] = normalized

    return {
        **state,
        "interview_questions": interview_questions,
        "rag_context_questions": rag_context_questions,
        "current_step": "checkpoint_2",
    }