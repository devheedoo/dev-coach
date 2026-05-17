"""Dev Coach: LangGraph interview coach with spaced repetition and SQLite persistence."""

from __future__ import annotations

import difflib
import os
import time
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import OpenAI
from pydantic import BaseModel, Field

import dev_coach_storage as storage

load_dotenv()

_KOREAN_OUTPUT_RULE = (
    "언어 규칙(필수): 사용자에게 보이는 모든 출력은 한국어로만 작성합니다. "
    "면접 질문·평가 피드백·꼬리 질문·학습 주제 라벨(topic_label)에 영어 문장을 쓰지 마세요. "
    "라이브러리 이름·API·코드 식별자처럼 업계에서 통용되는 고유명사는 필요 최소한만 포함할 수 있습니다.\n"
)


# --- Structured LLM outputs ---


class GeneratedQuestion(BaseModel):
    """A single technical interview question tailored to the candidate."""

    question: str = Field(
        ...,
        min_length=10,
        description="단 하나의 기술 면접 질문. 반드시 한국어 문장만 사용.",
    )


class EvaluationResult(BaseModel):
    """Understanding-focused rubric; 10 means confident mastery for each axis."""

    accuracy: int = Field(..., ge=1, le=10, description="기술적 정확성 점수(10이면 완전히 정확).")
    depth: int = Field(..., ge=1, le=10, description="설명의 깊이(10이면 깊고 균형 잡힌 설명).")
    practical_experience: int = Field(
        ...,
        ge=1,
        le=10,
        description="실무 경험 근거(10이면 구체적 근거가 매우 충분함).",
    )
    communication: int = Field(
        ...,
        ge=1,
        le=10,
        description="명확성과 구조(10이면 매우 명확하고 잘 정리됨).",
    )
    feedback: str = Field(
        ...,
        description="긍정적이고 건설적인 피드백. 한국어만 사용. 강점 후 보완 포인트.",
    )
    exemplary_answer: str = Field(
        ...,
        description=(
            "위 질문에 대한 모범적인 답안(개념 정리·구조·예시 중 필요한 것). "
            "한국어, 대략 8~12줄 분량(너무 길지 않게)."
        ),
    )


class LearningFocusTopic(BaseModel):
    """A topic label paired with how confidently the candidate currently understands it."""

    topic_label: str = Field(..., description="짧은 학습 주제 명사구. 한국어만.")
    understanding_confidence_score_1_to_10: int = Field(
        ...,
        ge=1,
        le=10,
        description="해당 주제에 대한 이해·자신감(10이면 충분히 숙달한 것으로 보임).",
    )


class LearningFocusExtraction(BaseModel):
    """Topics worth revisiting to deepen understanding (positive framing)."""

    learning_focus_topics: list[LearningFocusTopic] = Field(
        default_factory=list,
        description="추가 연습이 이해도·자신감을 높일 주제 2~5개. 라벨은 한국어.",
    )


class FollowUpQuestion(BaseModel):
    """One targeted follow-up based on evaluation and learning focus topics."""

    follow_up_question: str = Field(
        ...,
        min_length=10,
        description="단 하나의 꼬리 질문. 반드시 한국어로만 작성.",
    )


# --- Graph state ---


class DevCoachState(TypedDict, total=False):
    """Conversation state for one interview turn."""

    profile: NotRequired[dict[str, Any]]
    topic: NotRequired[str]
    learner_identifier: NotRequired[str]
    prefer_new_topic: NotRequired[bool]

    review_cards: list[dict[str, Any]]
    tracked_learning_topic_labels: list[str]
    new_learning_focus_topics: list[dict[str, Any]]

    active_review_topic_label: NotRequired[str]
    question_origin: NotRequired[str]
    current_question_topic_label: NotRequired[str]

    current_question: str
    user_answer: str
    evaluation: dict[str, Any]
    follow_up_question: str
    follow_up_answer: NotRequired[str]
    """Latest 꼬리 질문에 대한 답변(히스토리는 follow_up_history에 누적)."""

    follow_up_history: NotRequired[list[dict[str, Any]]]
    """완료된 꼬리 Q&A·평가. 각 항목: round, question, answer, evaluation."""


def _llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY is not set. Export it before running the graph."
        raise RuntimeError(msg)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.4)


def _learner_identifier(state: DevCoachState) -> str:
    env_default = os.getenv("DEV_COACH_LEARNER_ID", "default")
    raw = state.get("learner_identifier")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return env_default


def _profile_text(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "제공된 프로필이 없습니다."
    lines = []
    for key in ("years_experience", "target_company", "role", "level", "notes"):
        if profile.get(key):
            lines.append(f"- {key}: {profile[key]}")
    return "\n".join(lines) if lines else str(profile)


def _normalize_question_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _is_follow_up_too_similar(
    new_q: str,
    main_question: str,
    history: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.88,
) -> bool:
    """True if the candidate follow-up is empty, or too close to the main or a prior follow-up."""
    if not str(new_q).strip():
        return True
    n = _normalize_question_text(new_q)
    if not n:
        return True
    main_n = _normalize_question_text(main_question)
    if main_n and (
        n == main_n
        or difflib.SequenceMatcher(None, n, main_n).ratio() >= similarity_threshold
    ):
        return True
    for item in history:
        prev = _normalize_question_text(str(item.get("question", "")))
        if prev and (
            n == prev
            or difflib.SequenceMatcher(None, n, prev).ratio() >= similarity_threshold
        ):
            return True
    return False


def map_rubric_average_to_review_quality_score_0_to_5(average_1_to_10: float) -> int:
    """Maps rubric average (1-10) onto an integer quality score used only for scheduling."""
    clamped = max(1.0, min(10.0, float(average_1_to_10)))
    return int(round((clamped - 1.0) / 9.0 * 5.0))


def average_understanding_confidence_score_1_to_10(evaluation: dict[str, Any]) -> int:
    """Derive a conservative confidence score from the rubric axes."""
    axis_keys = ("accuracy", "depth", "practical_experience", "communication")
    values: list[float] = []
    for key in axis_keys:
        value = evaluation.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 5
    return int(round(max(1.0, min(10.0, sum(values) / len(values)))))


def fallback_question_topic_label(question: str, default_label: str = "") -> str:
    """Create a deterministic card label if structured extraction returns nothing."""
    cleaned = " ".join(str(question).strip().replace("\n", " ").split())
    if cleaned:
        if len(cleaned) > 42:
            cleaned = cleaned[:42].rstrip() + "..."
        return cleaned
    return default_label.strip() or "일반 소프트웨어 엔지니어링"


def default_review_card(topic_label: str) -> dict[str, Any]:
    """Creates a new spaced-repetition card with conservative defaults."""
    label = topic_label.strip()
    return {
        "topic_label": label,
        "memory_easiness_factor": 2.5,
        "successful_repetition_count": 0,
        "interval_until_next_review_days": 1.0,
        "next_review_unix_timestamp": 0.0,
        "last_understanding_confidence_score_1_to_10": None,
    }


def apply_spacing_after_review_attempt(
    card: dict[str, Any],
    review_quality_score_0_to_5: int,
    now_unix: float,
) -> dict[str, Any]:
    """
    Updates spacing metadata after a scheduled review attempt.

    Uses the widely published spaced repetition update rules where intervals expand after
    successful recalls and shrink after difficult recalls; ease-of-memory factor never drops
    below 1.3.
    """
    updated = dict(card)
    memory_easiness_factor = float(updated.get("memory_easiness_factor", 2.5))
    successful_repetition_count = int(updated.get("successful_repetition_count", 0))
    prior_interval_days = float(updated.get("interval_until_next_review_days", 1.0))

    if review_quality_score_0_to_5 < 3:
        updated["successful_repetition_count"] = 0
        interval_days = 1.0
        updated["interval_until_next_review_days"] = interval_days
        updated["next_review_unix_timestamp"] = now_unix + interval_days * 86400.0
        updated["memory_easiness_factor"] = memory_easiness_factor
        return updated

    delta_quality = 5 - review_quality_score_0_to_5
    memory_easiness_factor = memory_easiness_factor + (
        0.1 - delta_quality * (0.08 + delta_quality * 0.02)
    )
    memory_easiness_factor = max(1.3, memory_easiness_factor)
    updated["memory_easiness_factor"] = memory_easiness_factor

    successful_repetition_count += 1
    updated["successful_repetition_count"] = successful_repetition_count

    if successful_repetition_count == 1:
        interval_days = 1.0
    elif successful_repetition_count == 2:
        interval_days = 6.0
    else:
        interval_days = max(1.0, float(round(prior_interval_days * memory_easiness_factor)))

    updated["interval_until_next_review_days"] = float(interval_days)
    updated["next_review_unix_timestamp"] = now_unix + interval_days * 86400.0
    return updated


def extract_response_output_text(response: Any) -> str:
    """Pulls plain assistant text out of an OpenAI Responses API payload."""
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def gather_web_research_notes(topic_label: str, profile_text: str) -> str:
    """
    Uses OpenAI hosted web search via the Responses API.

    Falls back to an empty string when credentials are missing or the request fails.
    Model defaults follow OPENAI_RESEARCH_MODEL then OPENAI_MODEL.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or os.getenv("DEV_COACH_DISABLE_WEB_SEARCH") == "1":
        return ""

    research_model = os.getenv("OPENAI_RESEARCH_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=research_model,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=(
                "당신은 면접 코치를 돕습니다.\n"
                "웹 검색은 최신성에 도움이 될 때만 사용하세요.\n"
                "주제와 관련된 요약을 최대 8줄 이내 한글 불릿으로만 작성하세요.\n"
                f"주제 라벨: {topic_label}\n"
                f"후보자 프로필:\n{profile_text}\n"
            ),
        )
        return extract_response_output_text(response)
    except Exception:
        return ""


def load_review_cards_from_sqlite(state: DevCoachState) -> dict[str, Any]:
    """Hydrate spaced repetition cards from SQLite for this learner."""
    learner = _learner_identifier(state)
    cards = storage.load_review_cards(learner)
    labels = sorted({str(c["topic_label"]).strip() for c in cards if str(c.get("topic_label", "")).strip()})
    return {
        "learner_identifier": learner,
        "review_cards": cards,
        "tracked_learning_topic_labels": labels,
    }


def pick_review_or_new_topic(state: DevCoachState) -> dict[str, Any]:
    """Chooses whether this turn should run a scheduled review or explicit topic question."""
    topic = (state.get("topic") or "").strip()
    if state.get("prefer_new_topic") and topic:
        return {
            "question_origin": "new_topic",
            "active_review_topic_label": "",
        }

    now = time.time()
    cards = list(state.get("review_cards") or [])
    due: list[dict[str, Any]] = []
    for card in cards:
        try:
            if float(card.get("next_review_unix_timestamp", 0.0)) <= now:
                due.append(card)
        except (TypeError, ValueError):
            continue

    if due:
        due.sort(key=lambda c: float(c.get("next_review_unix_timestamp", 0.0)))
        label = str(due[0].get("topic_label", "")).strip()
        return {
            "question_origin": "scheduled_review",
            "active_review_topic_label": label,
        }

    return {
        "question_origin": "new_topic",
        "active_review_topic_label": "",
    }


def route_after_pick(state: DevCoachState) -> Literal["scheduled_review_question", "new_topic_question"]:
    return (
        "scheduled_review_question"
        if state.get("question_origin") == "scheduled_review"
        else "new_topic_question"
    )


def generate_question(state: DevCoachState) -> dict[str, Any]:
    """Create the main interview question from optional profile and topic."""
    llm = _llm().with_structured_output(GeneratedQuestion)
    topic = (state.get("topic") or "general software engineering").strip()
    profile = state.get("profile")
    prompt = (
        _KOREAN_OUTPUT_RULE
        + "당신은 경험 많은 소프트웨어 엔지니어링 면접관입니다.\n"
        "기술 면접 질문을 정확히 하나만 만드세요.\n\n"
        f"주제 초점: {topic}\n"
        f"후보자 프로필:\n{_profile_text(profile)}\n\n"
        "요구사항:\n"
        "- 프로필이 알려지면 난이도를 맞추고, 없으면 중간 난이도를 기본으로 합니다.\n"
        "- 깊이를 끌어낼 수 있는 개방형 질문을 선호합니다.\n"
        "- 기대치를 긍정적으로 제시합니다.\n"
        "- 인공지능이라고 밝히지 마세요.\n"
    )
    out: GeneratedQuestion = llm.invoke(prompt)
    return {
        "current_question": out.question.strip(),
        "active_review_topic_label": "",
        "question_origin": "new_topic",
        "current_question_topic_label": topic,
    }


def generate_scheduled_review_question(state: DevCoachState) -> dict[str, Any]:
    """Ask a review question for a due topic, optionally grounding notes via OpenAI web search."""
    topic = (state.get("active_review_topic_label") or "").strip()
    profile = state.get("profile")
    profile_text = _profile_text(profile)
    research_notes = gather_web_research_notes(topic, profile_text)

    llm = _llm().with_structured_output(GeneratedQuestion)
    prompt_parts = [
        _KOREAN_OUTPUT_RULE,
        "예약된 복습 차례입니다. 아래 주제를 다시 점검하는 기술 면접 질문을 정확히 하나만 만드세요.",
        "정의 너머의 깊이와, 필요하면 구체적 사례를 요청하세요.",
        "존중하고 자신감을 북돋는 톤을 유지하세요.\n",
        f"복습 주제 라벨: {topic}",
        f"후보자 프로필:\n{profile_text}",
    ]
    if research_notes:
        prompt_parts.extend(["\n참고 조사 노트(한글):\n", research_notes])

    out: GeneratedQuestion = llm.invoke("\n".join(prompt_parts))
    return {
        "current_question": out.question.strip(),
        "question_origin": "scheduled_review",
        "current_question_topic_label": topic,
    }


def collect_answer(state: DevCoachState) -> dict[str, Any]:
    """Pause for human answer (Human-in-the-loop via interrupt)."""
    payload = {
        "type": "user_answer",
        "question": state.get("current_question", ""),
        "hint": "재개할 때 답변 문자열을 넣거나 {'answer': '...'} 형식의 JSON을 전달하세요.",
    }
    raw = interrupt(payload)

    if isinstance(raw, str):
        answer_text = raw.strip()
    elif isinstance(raw, dict):
        answer_text = str(raw.get("answer", raw)).strip()
    else:
        answer_text = str(raw).strip()

    if not answer_text:
        answer_text = "(빈 답변)"

    return {"user_answer": answer_text}


def evaluate_answer(state: DevCoachState) -> dict[str, Any]:
    """Score the answer with a 1-10 understanding-focused rubric."""
    llm = _llm().with_structured_output(EvaluationResult)
    q = state.get("current_question", "")
    a = state.get("user_answer", "")
    prompt = (
        _KOREAN_OUTPUT_RULE
        + "아래 면접 답변을 루브릭 필드로 평가하세요.\n"
        "각 축은 1~10점이며 10은 해당 영역에서 완전히 이해하고 자신 있게 설명한 수준입니다.\n"
        "불완전하지만 방향이 맞는 답에는 공정하게 부분 점수를 주세요.\n"
        "피드백(feedback)은 반드시 한국어로, 먼저 강점을 요약한 뒤 구체적 성장 포인트를 적으세요.\n"
        "exemplary_answer는 위 질문에 대해 모범적으로 구성된 답변으로, 약 8~12줄(짧은 문단 또는 불릿 혼합)로 작성하세요. "
        "질문이 요구하는 핵심을 빠짐없이 짚되, 후보 답변을 베끼지 말고 새로 서술합니다.\n\n"
        f"질문:\n{q}\n\n"
        f"답변:\n{a}\n"
    )
    out: EvaluationResult = llm.invoke(prompt)
    return {
        "evaluation": {
            "accuracy": out.accuracy,
            "depth": out.depth,
            "practical_experience": out.practical_experience,
            "communication": out.communication,
            "feedback": out.feedback,
            "exemplary_answer": out.exemplary_answer,
        }
    }


def collect_follow_up_answer(state: DevCoachState) -> dict[str, Any]:
    """Pause for the answer to the current follow-up question."""
    history = list(state.get("follow_up_history") or [])
    round_index = len(history) + 1
    payload = {
        "type": "follow_up_answer",
        "round": round_index,
        "question": state.get("follow_up_question", ""),
        "hint": "재개할 때 답변 문자열을 넣거나 {'answer': '...'} 형식의 JSON을 전달하세요.",
    }
    raw = interrupt(payload)

    if isinstance(raw, str):
        answer_text = raw.strip()
    elif isinstance(raw, dict):
        answer_text = str(raw.get("answer", raw)).strip()
    else:
        answer_text = str(raw).strip()

    if not answer_text:
        answer_text = "(빈 답변)"

    return {"follow_up_answer": answer_text}


def evaluate_follow_up_answer(state: DevCoachState) -> dict[str, Any]:
    """Score the follow-up answer with the same rubric as the main answer."""
    llm = _llm().with_structured_output(EvaluationResult)
    q = state.get("follow_up_question", "")
    a = state.get("follow_up_answer", "")
    prompt = (
        _KOREAN_OUTPUT_RULE
        + "아래는 꼬리 질문에 대한 답변입니다. 루브릭 필드로 평가하세요.\n"
        "각 축은 1~10점이며 10은 해당 영역에서 완전히 이해하고 자신 있게 설명한 수준입니다.\n"
        "불완전하지만 방향이 맞는 답에는 공정하게 부분 점수를 주세요.\n"
        "피드백(feedback)은 반드시 한국어로, 먼저 강점을 요약한 뒤 구체적 성장 포인트를 적으세요.\n"
        "exemplary_answer는 **해당 꼬리 질문**에 대해 모범적으로 구성된 답으로, 약 8~12줄로 작성하세요. "
        "후보 답변을 베끼지 말고 새로 서술합니다.\n\n"
        f"꼬리 질문:\n{q}\n\n"
        f"답변:\n{a}\n"
    )
    out: EvaluationResult = llm.invoke(prompt)
    eval_dict = {
        "accuracy": out.accuracy,
        "depth": out.depth,
        "practical_experience": out.practical_experience,
        "communication": out.communication,
        "feedback": out.feedback,
        "exemplary_answer": out.exemplary_answer,
    }
    history = list(state.get("follow_up_history") or [])
    round_index = len(history) + 1
    history.append(
        {
            "round": round_index,
            "question": q.strip(),
            "answer": a.strip(),
            "evaluation": eval_dict,
        }
    )
    return {"follow_up_history": history}


def update_review_schedule_after_evaluation(state: DevCoachState) -> dict[str, Any]:
    """Advance spaced repetition metadata after a scheduled review question."""
    if state.get("question_origin") != "scheduled_review":
        return {}

    evaluation = state.get("evaluation") or {}
    axis_keys = ("accuracy", "depth", "practical_experience", "communication")
    axis_values: list[float] = []
    for key in axis_keys:
        value = evaluation.get(key)
        if isinstance(value, (int, float)):
            axis_values.append(float(value))
    average_score = sum(axis_values) / len(axis_values) if axis_values else 5.5
    review_quality_score = map_rubric_average_to_review_quality_score_0_to_5(average_score)

    label = (state.get("active_review_topic_label") or "").strip()
    cards = list(state.get("review_cards") or [])
    now = time.time()

    updated_cards: list[dict[str, Any]] = []
    matched = False
    for card in cards:
        topic = str(card.get("topic_label", "")).strip()
        if topic == label:
            matched = True
            updated_cards.append(apply_spacing_after_review_attempt(dict(card), review_quality_score, now))
        else:
            updated_cards.append(dict(card))

    if label and not matched:
        seed = default_review_card(label)
        updated_cards.append(apply_spacing_after_review_attempt(seed, review_quality_score, now))

    return {
        "review_cards": updated_cards,
        "active_review_topic_label": "",
    }


def analyze_learning_focus(state: DevCoachState) -> dict[str, Any]:
    """Extract review-card topics from the main question and feedback."""
    llm = _llm().with_structured_output(LearningFocusExtraction)
    question = state.get("current_question", "")
    answer = state.get("user_answer", "")
    evaluation = state.get("evaluation") or {}
    feedback = evaluation.get("feedback", "")
    exemplary = evaluation.get("exemplary_answer", "")
    prompt = (
        _KOREAN_OUTPUT_RULE
        + "아래 면접 질문·답변·평가에서 review_cards에 저장할 학습 주제를 2~5개 추출하세요.\n"
        "중요: 첫 번째 learning_focus_topics 항목은 반드시 **이 질문 자체의 핵심 주제**여야 합니다.\n"
        "첫 번째 topic_label은 '첫 번째 질문', '원본 질문' 같은 위치 표현을 쓰지 말고, "
        "복습 카드로 다시 물어볼 수 있는 구체적인 개념/상황 명사구로 작성하세요.\n"
        "나머지 항목은 피드백에서 드러난 보완 주제를 추가하세요.\n"
        "각 topic_label은 짧은 한글 명사구여야 하며, 질문마다 초점이 다르면 서로 다른 라벨로 구분하세요.\n"
        "understanding_confidence_score_1_to_10은 해당 주제에서 후보가 보여 준 이해·자신감 수준입니다(10이면 매우 충분함).\n"
        "전향적이고 건설적인 표현을 쓰세요.\n\n"
        f"질문:\n{question}\n\n"
        f"답변:\n{answer}\n\n"
        f"피드백:\n{feedback}\n"
        f"모범 답안:\n{exemplary}\n"
    )
    out: LearningFocusExtraction = llm.invoke(prompt)
    topics = [item.model_dump() for item in out.learning_focus_topics]
    if not topics:
        topics = [
            {
                "topic_label": fallback_question_topic_label(
                    question,
                    str(state.get("current_question_topic_label", "")),
                ),
                "understanding_confidence_score_1_to_10": average_understanding_confidence_score_1_to_10(
                    evaluation
                ),
            }
        ]
    return {"new_learning_focus_topics": topics}


def analyze_follow_up_learning_focus(state: DevCoachState) -> dict[str, Any]:
    """Extract review-card topics from the latest follow-up question and evaluation."""
    history = list(state.get("follow_up_history") or [])
    if not history:
        return {"new_learning_focus_topics": []}
    last_item = history[-1]
    question = str(last_item.get("question", ""))
    answer = str(last_item.get("answer", ""))
    last_eval = history[-1].get("evaluation") or {}
    feedback = last_eval.get("feedback", "")
    exemplary = last_eval.get("exemplary_answer", "")
    llm = _llm().with_structured_output(LearningFocusExtraction)
    prompt = (
        _KOREAN_OUTPUT_RULE
        + "아래 꼬리 질문·답변·평가에서 review_cards에 저장할 학습 주제를 2~5개 추출하세요.\n"
        "중요: 첫 번째 learning_focus_topics 항목은 반드시 **이 꼬리 질문 자체의 핵심 주제**여야 합니다.\n"
        "첫 번째 topic_label은 '꼬리 질문', '두 번째 질문' 같은 위치 표현을 쓰지 말고, "
        "복습 카드로 다시 물어볼 수 있는 구체적인 개념/상황 명사구로 작성하세요.\n"
        "나머지 항목은 피드백에서 드러난 보완 주제를 추가하세요.\n"
        "각 topic_label은 짧은 한글 명사구여야 하며, 질문마다 초점이 다르면 서로 다른 라벨로 구분하세요.\n"
        "understanding_confidence_score_1_to_10은 해당 주제에서 후보가 보여 준 이해·자신감 수준입니다(10이면 매우 충분함).\n"
        "전향적이고 건설적인 표현을 쓰세요.\n\n"
        f"꼬리 질문:\n{question}\n\n"
        f"답변:\n{answer}\n\n"
        f"피드백:\n{feedback}\n"
        f"모범 답안:\n{exemplary}\n"
    )
    out: LearningFocusExtraction = llm.invoke(prompt)
    topics = [item.model_dump() for item in out.learning_focus_topics]
    if not topics:
        topics = [
            {
                "topic_label": fallback_question_topic_label(question),
                "understanding_confidence_score_1_to_10": average_understanding_confidence_score_1_to_10(
                    last_eval
                ),
            }
        ]
    return {"new_learning_focus_topics": topics}


def _merge_learning_topics_impl(state: DevCoachState) -> dict[str, Any]:
    """Merge structured learning topics into review cards and tracked labels."""
    cards_by_topic: dict[str, dict[str, Any]] = {}
    for card in state.get("review_cards") or []:
        label = str(card.get("topic_label", "")).strip()
        if label:
            cards_by_topic[label] = dict(card)

    for item in state.get("new_learning_focus_topics") or []:
        label = str(item.get("topic_label", "")).strip()
        score = item.get("understanding_confidence_score_1_to_10")
        if not label:
            continue
        if label not in cards_by_topic:
            cards_by_topic[label] = default_review_card(label)
        if isinstance(score, int):
            cards_by_topic[label]["last_understanding_confidence_score_1_to_10"] = score

    merged_cards = list(cards_by_topic.values())
    merged_cards.sort(key=lambda c: str(c.get("topic_label", "")))
    labels = sorted(cards_by_topic.keys())

    return {
        "review_cards": merged_cards,
        "tracked_learning_topic_labels": labels,
        "new_learning_focus_topics": [],
    }


def merge_main_learning_topics(state: DevCoachState) -> dict[str, Any]:
    """Merge topics extracted from the main answer evaluation into review cards."""
    return _merge_learning_topics_impl(state)


def merge_follow_up_learning_topics(state: DevCoachState) -> dict[str, Any]:
    """Merge topics extracted from the latest follow-up answer evaluation."""
    return _merge_learning_topics_impl(state)


def persist_review_cards_sqlite(state: DevCoachState) -> dict[str, Any]:
    """Persist the latest review card snapshot for the learner."""
    learner = _learner_identifier(state)
    storage.replace_all_review_cards(learner, list(state.get("review_cards") or []))
    return {}


def route_after_follow_up_merge(
    state: DevCoachState,
) -> Literal["more_follow_up", "persist"]:
    """After three completed follow-up rounds, persist; otherwise continue the follow-up chain."""
    history = state.get("follow_up_history") or []
    if len(history) < 3:
        return "more_follow_up"
    return "persist"


def generate_follow_up(state: DevCoachState) -> dict[str, Any]:
    """Produce the next follow-up question (up to 3 rounds), using prior rounds for context."""
    history = list(state.get("follow_up_history") or [])
    next_round = len(history) + 1
    if next_round > 3:
        return {}

    llm = _llm().with_structured_output(FollowUpQuestion)
    main_q = state.get("current_question", "")
    a = state.get("user_answer", "")
    evaluation = state.get("evaluation") or {}
    topics = state.get("tracked_learning_topic_labels") or []

    prior_lines: list[str] = []
    for item in history:
        r = item.get("round", "?")
        pq = item.get("question", "")
        pa = item.get("answer", "")
        pe = item.get("evaluation") or {}
        prior_lines.append(
            f"--- {r}번째 꼬리 질문 ---\n질문: {pq}\n답변: {pa}\n점수 요약: {pe}\n"
        )
    prior_block = "\n".join(prior_lines) if prior_lines else "(이전 꼬리 질문 없음)"

    base_prompt = (
        _KOREAN_OUTPUT_RULE
        + f"면접관입니다. 지금은 꼬리 질문의 **{next_round}번째**입니다(최대 3번).\n"
        "**필수:** 출력 필드 `follow_up_question`에는 **반드시 짧은 질문 문장 하나만** 넣으세요. "
        "원본 질문 전문을 인용·복사·붙여넣기 하지 마세요.\n"
        "꼬리 질문은 원본 질문과 **문장이 같거나 거의 같아서는 안 됩니다**. "
        "첫 답변·이전 꼬리 답변에서 덜 드러난 **한 가지 측면**(원인, 경계 조건, 트레이드오프, 검증, 운영 등)을 "
        "좁혀서 **새로운 초점**으로 물으세요.\n"
        "이전 꼬리 질문과도 **똑같은 문구**로 반복하지 마세요.\n\n"
        f"원본 질문:\n{main_q}\n\n"
        f"후보의 첫 답변:\n{a}\n\n"
        f"첫 답변에 대한 점수·피드백·모범답 요약(참고만):\n{evaluation}\n\n"
        f"완료된 꼬리 질문·답변:\n{prior_block}\n\n"
        f"학습 포커스 주제 라벨: {', '.join(topics) if topics else '(없음)'}\n\n"
    )
    retry_hint = (
        "\n**재시도 이유:** 직전에 만든 꼬리 질문이 원본·이전 꼬리와 너무 비슷했습니다. "
        "완전히 다른 관점에서 질문을 다시 만드세요.\n"
    )

    candidate = ""
    for attempt in range(3):
        prompt = base_prompt + (retry_hint if attempt else "")
        out: FollowUpQuestion = llm.invoke(prompt)
        candidate = out.follow_up_question.strip()
        if not _is_follow_up_too_similar(candidate, main_q, history):
            break

    return {"follow_up_question": candidate}


def build_graph(*, checkpointer=None):
    builder = StateGraph(DevCoachState)
    builder.add_node("load_review_cards_from_sqlite", load_review_cards_from_sqlite)
    builder.add_node("pick_review_or_new_topic", pick_review_or_new_topic)
    builder.add_node("generate_question", generate_question)
    builder.add_node("generate_scheduled_review_question", generate_scheduled_review_question)
    builder.add_node("collect_answer", collect_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("update_review_schedule_after_evaluation", update_review_schedule_after_evaluation)
    builder.add_node("analyze_learning_focus", analyze_learning_focus)
    builder.add_node("merge_main_learning_topics", merge_main_learning_topics)
    builder.add_node("persist_review_cards_sqlite", persist_review_cards_sqlite)
    builder.add_node("generate_follow_up", generate_follow_up)
    builder.add_node("collect_follow_up_answer", collect_follow_up_answer)
    builder.add_node("evaluate_follow_up_answer", evaluate_follow_up_answer)
    builder.add_node("analyze_follow_up_learning_focus", analyze_follow_up_learning_focus)
    builder.add_node("merge_follow_up_learning_topics", merge_follow_up_learning_topics)

    builder.add_edge(START, "load_review_cards_from_sqlite")
    builder.add_edge("load_review_cards_from_sqlite", "pick_review_or_new_topic")
    builder.add_conditional_edges(
        "pick_review_or_new_topic",
        route_after_pick,
        {
            "scheduled_review_question": "generate_scheduled_review_question",
            "new_topic_question": "generate_question",
        },
    )
    builder.add_edge("generate_question", "collect_answer")
    builder.add_edge("generate_scheduled_review_question", "collect_answer")
    builder.add_edge("collect_answer", "evaluate_answer")

    builder.add_edge("evaluate_answer", "update_review_schedule_after_evaluation")
    builder.add_edge("evaluate_answer", "analyze_learning_focus")
    builder.add_edge(
        ["update_review_schedule_after_evaluation", "analyze_learning_focus"],
        "merge_main_learning_topics",
    )
    builder.add_edge("merge_main_learning_topics", "generate_follow_up")
    builder.add_edge("generate_follow_up", "collect_follow_up_answer")
    builder.add_edge("collect_follow_up_answer", "evaluate_follow_up_answer")
    builder.add_edge("evaluate_follow_up_answer", "analyze_follow_up_learning_focus")
    builder.add_edge("analyze_follow_up_learning_focus", "merge_follow_up_learning_topics")
    builder.add_conditional_edges(
        "merge_follow_up_learning_topics",
        route_after_follow_up_merge,
        {
            "more_follow_up": "generate_follow_up",
            "persist": "persist_review_cards_sqlite",
        },
    )
    builder.add_edge("persist_review_cards_sqlite", END)

    # LangGraph Platform / `langgraph dev` injects persistence when deployed there.
    # For notebooks/tests you may pass a MemorySaver instance via `checkpointer`.
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
