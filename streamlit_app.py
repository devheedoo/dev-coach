"""Streamlit chat UI for Dev Coach (LangGraph `build_graph` + MemorySaver)."""

from __future__ import annotations

import os
import uuid
import warnings
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from dev_coach import build_graph

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change",
    category=DeprecationWarning,
)


def _ensure_graph() -> Any:
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(checkpointer=MemorySaver())
    return st.session_state.graph


def _init_message_list() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _question_origin_label(origin: str | None) -> str:
    if origin == "scheduled_review":
        return "예약 복습"
    if origin == "new_topic":
        return "입력 주제"
    return "알 수 없음"


def _format_question_context_markdown(result: dict[str, Any]) -> str:
    topic = (result.get("current_question_topic_label") or "").strip()
    origin = _question_origin_label(result.get("question_origin"))
    if topic:
        return f"**사용된 주제:** `{topic}`  \n**질문 출처:** {origin}"
    return f"**질문 출처:** {origin}"


def _format_evaluation_markdown(result: dict[str, Any]) -> str:
    evaluation = result.get("evaluation") or {}
    accuracy = evaluation.get("accuracy")
    depth = evaluation.get("depth")
    practical = evaluation.get("practical_experience")
    comm = evaluation.get("communication")
    feedback = evaluation.get("feedback") or ""

    topics = result.get("tracked_learning_topic_labels") or []
    follow = (result.get("follow_up_question") or "").strip()

    lines: list[str] = []
    lines.append("### 평가 요약")
    lines.append(_format_question_context_markdown(result))
    lines.append(
        f"- 기술적 정확성: **{accuracy}**/10\n"
        f"- 설명 깊이: **{depth}**/10\n"
        f"- 실무 근거: **{practical}**/10\n"
        f"- 명확성·구조: **{comm}**/10"
    )
    lines.append(f"\n**피드백**\n\n{feedback}")
    if topics:
        lines.append(f"\n**학습 포커스 주제**\n\n{', '.join(str(t) for t in topics)}")
    if follow:
        lines.append(f"\n**꼬리 질문**\n\n{follow}")
    lines.append(
        "\n---\n*한 턴이 끝났습니다. 사이드바에서 **Start / Reset interview**로 새 질문을 시작하세요.*"
    )
    return "\n".join(lines)


def _start_interview(topic: str, learner_identifier: str, profile: dict[str, Any] | None) -> None:
    graph = _ensure_graph()
    _init_message_list()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.config = config
    st.session_state.waiting_for_answer = False
    st.session_state.messages = []

    payload: dict[str, Any] = {
        "topic": topic.strip() or "general software engineering",
        "learner_identifier": learner_identifier.strip() or os.getenv("DEV_COACH_LEARNER_ID", "default"),
        "prefer_new_topic": bool(topic.strip()),
    }
    if profile:
        payload["profile"] = profile

    try:
        result = graph.invoke(payload, config=config)
    except RuntimeError as exc:
        st.error(f"실행 오류: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"예기치 않은 오류: {exc}")
        return

    interrupts = result.get("__interrupt__")
    if interrupts:
        st.session_state.waiting_for_answer = True
        question = (result.get("current_question") or "").strip()
        if not question and interrupts:
            first = interrupts[0]
            if hasattr(first, "value") and isinstance(first.value, dict):
                question = str(first.value.get("question", "")).strip()
        if not question:
            question = "(질문을 불러오지 못했습니다. 콘솔 로그를 확인하세요.)"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"### 면접 질문\n\n{_format_question_context_markdown(result)}\n\n{question}",
            }
        )
    else:
        st.session_state.waiting_for_answer = False
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": _format_evaluation_markdown(result),
            }
        )


def _submit_answer(user_text: str) -> None:
    graph = _ensure_graph()
    config = st.session_state.get("config")
    if not config:
        st.error("먼저 사이드바에서 인터뷰를 시작하세요.")
        return

    st.session_state.messages.append({"role": "user", "content": user_text})

    try:
        result = graph.invoke(Command(resume={"answer": user_text}), config=config)
    except RuntimeError as exc:
        st.error(f"실행 오류: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"예기치 않은 오류: {exc}")
        return

    st.session_state.waiting_for_answer = False
    st.session_state.messages.append(
        {"role": "assistant", "content": _format_evaluation_markdown(result)}
    )


def main() -> None:
    st.set_page_config(page_title="Dev Coach", page_icon="💬", layout="centered")
    st.title("Dev Coach")
    st.caption("LangGraph agent와 채팅 형태로 면접 연습을 진행합니다.")

    _init_message_list()
    _ensure_graph()

    with st.sidebar:
        st.header("설정")
        topic = st.text_input("주제 (topic)", value="React performance")
        learner_id = st.text_input(
            "학습자 ID (learner_identifier)",
            value=os.getenv("DEV_COACH_LEARNER_ID", "streamlit_user"),
        )
        st.subheader("프로필 (선택)")
        years = st.text_input("경력 년수 (years_experience)", value="5")
        target_company = st.text_input("목표 회사 (target_company)", value="")
        role = st.text_input("역할 (role)", value="Frontend Engineer")
        level = st.text_input("레벨 (level)", value="Senior")
        notes = st.text_area("노트 (notes)", value="", height=68)

        profile: dict[str, Any] = {}
        if years.strip():
            try:
                profile["years_experience"] = int(years.strip())
            except ValueError:
                profile["years_experience"] = years.strip()
        if target_company.strip():
            profile["target_company"] = target_company.strip()
        if role.strip():
            profile["role"] = role.strip()
        if level.strip():
            profile["level"] = level.strip()
        if notes.strip():
            profile["notes"] = notes.strip()

        if st.button("Start / Reset interview", type="primary"):
            _start_interview(topic, learner_id, profile if profile else None)

        st.divider()
        st.caption("`OPENAI_API_KEY`가 필요합니다. SQLite 경로는 `DEV_COACH_SQLITE_PATH`로 바꿀 수 있습니다.")

    waiting = bool(st.session_state.get("waiting_for_answer"))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "답변을 입력하세요…" if waiting else "사이드바에서 인터뷰를 시작하면 답변을 입력할 수 있습니다.",
        disabled=not waiting,
    )
    if prompt and waiting:
        _submit_answer(prompt.strip())
        st.rerun()


if __name__ == "__main__":
    main()
