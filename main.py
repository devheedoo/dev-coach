"""Dev Coach v1: LangGraph interview coach (question, evaluate, follow-up, weakness memory)."""

from __future__ import annotations

import os
from typing import Any, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

load_dotenv()


# --- Structured LLM outputs ---


class GeneratedQuestion(BaseModel):
    """A single technical interview question tailored to the candidate."""

    question: str = Field(
        ...,
        min_length=10,
        description="Clear interview question in English or Korean per user preference in profile.",
    )


class EvaluationResult(BaseModel):
    """Rubric aligned with README: accuracy, depth, practical experience, clarity."""

    accuracy: int = Field(..., ge=1, le=5, description="Technical correctness.")
    depth: int = Field(..., ge=1, le=5, description="Depth of explanation.")
    practical_experience: int = Field(
        ..., ge=1, le=5, description="Evidence of real-world experience."
    )
    communication: int = Field(..., ge=1, le=5, description="Clarity and structure.")
    feedback: str = Field(
        ...,
        description="Concise feedback: what was good, what to improve, with specifics.",
    )


class WeaknessExtraction(BaseModel):
    """Short topic tags the candidate should review (weakness memory)."""

    weaknesses: list[str] = Field(
        default_factory=list,
        description="2-5 short topic labels, e.g. 'React reconciliation', 'HTTP caching'.",
    )


class FollowUpQuestion(BaseModel):
    """One targeted follow-up based on evaluation and weaknesses."""

    follow_up_question: str = Field(
        ...,
        min_length=10,
        description="A single follow-up that digs deeper into the weakest area.",
    )


# --- Graph state ---


class DevCoachState(TypedDict, total=False):
    """Conversation state for one interview turn (v1)."""

    profile: NotRequired[dict[str, Any]]
    topic: NotRequired[str]
    current_question: str
    user_answer: str
    evaluation: dict[str, Any]
    new_weaknesses: list[str]
    weaknesses: list[str]
    follow_up_question: str


def _llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY is not set. Export it before running the graph."
        raise RuntimeError(msg)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.4)


def _profile_text(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "No profile provided."
    lines = []
    for key in ("years_experience", "target_company", "role", "level", "notes"):
        if profile.get(key):
            lines.append(f"- {key}: {profile[key]}")
    return "\n".join(lines) if lines else str(profile)


def generate_question(state: DevCoachState) -> dict[str, Any]:
    """Create the main interview question from optional profile and topic."""
    llm = _llm().with_structured_output(GeneratedQuestion)
    topic = (state.get("topic") or "general software engineering").strip()
    profile = state.get("profile")
    prompt = (
        "You are an experienced engineering interviewer.\n"
        "Generate exactly ONE technical interview question.\n\n"
        f"Topic focus: {topic}\n"
        f"Candidate profile:\n{_profile_text(profile)}\n\n"
        "Requirements:\n"
        "- Match difficulty to the profile if known; otherwise default to mid-level.\n"
        "- Prefer open-ended questions that invite depth.\n"
        "- Do not mention that you are an AI.\n"
    )
    out: GeneratedQuestion = llm.invoke(prompt)
    return {"current_question": out.question.strip()}


def collect_answer(state: DevCoachState) -> dict[str, Any]:
    """Pause for human answer (Human-in-the-loop via interrupt)."""
    payload = {
        "type": "user_answer",
        "question": state.get("current_question", ""),
        "hint": "Resume the run with your answer text (string) or {'answer': '...'}.",
    }
    raw = interrupt(payload)

    if isinstance(raw, str):
        answer_text = raw.strip()
    elif isinstance(raw, dict):
        answer_text = str(raw.get("answer", raw)).strip()
    else:
        answer_text = str(raw).strip()

    if not answer_text:
        answer_text = "(empty answer)"

    return {"user_answer": answer_text}


def evaluate_answer(state: DevCoachState) -> dict[str, Any]:
    """Score the answer and produce feedback."""
    llm = _llm().with_structured_output(EvaluationResult)
    q = state.get("current_question", "")
    a = state.get("user_answer", "")
    prompt = (
        "Evaluate this interview answer using the rubric fields.\n"
        "Be fair: partial credit for incomplete but directionally correct answers.\n\n"
        f"QUESTION:\n{q}\n\n"
        f"ANSWER:\n{a}\n"
    )
    out: EvaluationResult = llm.invoke(prompt)
    return {
        "evaluation": {
            "accuracy": out.accuracy,
            "depth": out.depth,
            "practical_experience": out.practical_experience,
            "communication": out.communication,
            "feedback": out.feedback,
        }
    }


def analyze_weakness(state: DevCoachState) -> dict[str, Any]:
    """Derive weakness topic tags from the evaluation (for long-term-style memory in v1)."""
    llm = _llm().with_structured_output(WeaknessExtraction)
    ev = state.get("evaluation") or {}
    feedback = ev.get("feedback", "")
    prompt = (
        "From the interviewer feedback below, extract 2-5 SHORT weakness topic labels "
        "the candidate should study. Use noun phrases, no sentences.\n\n"
        f"FEEDBACK:\n{feedback}\n"
    )
    out: WeaknessExtraction = llm.invoke(prompt)
    tags = [w.strip() for w in out.weaknesses if w and str(w).strip()]
    return {"new_weaknesses": tags}


def store_weakness(state: DevCoachState) -> dict[str, Any]:
    """Merge new weaknesses into the accumulated weakness list in state."""
    existing = list(state.get("weaknesses") or [])
    new = list(state.get("new_weaknesses") or [])
    merged: list[str] = []
    for item in existing + new:
        t = item.strip()
        if t and t not in merged:
            merged.append(t)
    return {"weaknesses": merged, "new_weaknesses": []}


def generate_follow_up(state: DevCoachState) -> dict[str, Any]:
    """Produce one follow-up question targeting the weakest area."""
    llm = _llm().with_structured_output(FollowUpQuestion)
    q = state.get("current_question", "")
    a = state.get("user_answer", "")
    ev = state.get("evaluation") or {}
    ws = state.get("weaknesses") or []
    prompt = (
        "You are an interviewer. Create exactly ONE follow-up question.\n"
        "It must probe the weakest part of the answer or the weakness tags.\n\n"
        f"ORIGINAL_QUESTION:\n{q}\n\n"
        f"CANDIDATE_ANSWER:\n{a}\n\n"
        f"SCORES: {ev}\n"
        f"WEAKNESS_TAGS: {', '.join(ws) if ws else '(none)'}\n"
    )
    out: FollowUpQuestion = llm.invoke(prompt)
    return {"follow_up_question": out.follow_up_question.strip()}


def build_graph():
    builder = StateGraph(DevCoachState)
    builder.add_node("generate_question", generate_question)
    builder.add_node("collect_answer", collect_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("analyze_weakness", analyze_weakness)
    builder.add_node("store_weakness", store_weakness)
    builder.add_node("generate_follow_up", generate_follow_up)

    builder.add_edge(START, "generate_question")
    builder.add_edge("generate_question", "collect_answer")
    builder.add_edge("collect_answer", "evaluate_answer")
    builder.add_edge("evaluate_answer", "analyze_weakness")
    builder.add_edge("analyze_weakness", "store_weakness")
    builder.add_edge("store_weakness", "generate_follow_up")
    builder.add_edge("generate_follow_up", END)

    # LangGraph Platform / `langgraph dev` injects persistence; do not pass a custom checkpointer.
    return builder.compile()


graph = build_graph()


def main() -> None:
    print("Use `uv run langgraph dev` to run the Dev Coach graph (graph id: dev_coach).")


if __name__ == "__main__":
    main()
