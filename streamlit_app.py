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

from dev_coach import (
    STREAM_EVALUATION_CHUNK_KEY,
    STREAM_EVALUATION_PHASE_KEY,
    build_graph,
)

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


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """First LangGraph interrupt attachment as a dict, if present."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    val = getattr(first, "value", None)
    if val is None and isinstance(first, dict):
        val = first.get("value") or first.get("__value__")
    return val if isinstance(val, dict) else None


def _format_rubric_block_md(evaluation: dict[str, Any]) -> str:
    accuracy = evaluation.get("accuracy")
    depth = evaluation.get("depth")
    practical = evaluation.get("practical_experience")
    comm = evaluation.get("communication")
    feedback = evaluation.get("feedback") or ""
    exemplary = (evaluation.get("exemplary_answer") or "").strip()
    scores = (
        f"- 기술적 정확성: **{accuracy}**/10\n"
        f"- 설명 깊이: **{depth}**/10\n"
        f"- 실무 근거: **{practical}**/10\n"
        f"- 명확성·구조: **{comm}**/10"
    )
    parts: list[str] = [scores, f"**피드백**\n\n{feedback}"]
    if exemplary:
        parts.append(f"**모범 답안 참고**(질문에 대한 예시 답안)\n\n{exemplary}")
    return "\n\n".join(parts)


def _format_main_answer_evaluation_markdown(result: dict[str, Any]) -> str:
    """평가 블록: 원본(첫) 답변에 대한 루브릭."""
    evaluation = result.get("evaluation") or {}
    lines: list[str] = ["### 첫 답변 평가", _format_question_context_markdown(result), _format_rubric_block_md(evaluation)]
    return "\n\n".join(lines)


def _format_single_follow_up_evaluation_markdown(item: dict[str, Any]) -> str:
    rnd = item.get("round", "?")
    question = (item.get("question") or "").strip()
    evaluation = item.get("evaluation") or {}
    header = f"### 꼬리 질문 {rnd} 평가"
    q_line = f"**꼬리 질문 {rnd}**\n\n{question}" if question else ""
    body = _format_rubric_block_md(evaluation) if evaluation else ""
    parts = [p for p in (header, q_line, body) if p]
    return "\n\n".join(parts)


def _format_follow_up_step_markdown(result: dict[str, Any], interrupt_val: dict[str, Any]) -> str:
    """
    꼬리 질문에 답하기 전 단계: 첫 답변 평가 또는 직전 꼬리 답변 평가 + 다음 꼬리 질문."""
    history = list(result.get("follow_up_history") or [])
    fu_q = (interrupt_val.get("question") or result.get("follow_up_question") or "").strip()
    round_next = interrupt_val.get("round")
    if round_next is None:
        round_next = len(history) + 1
    try:
        round_next_i = int(round_next)
    except (TypeError, ValueError):
        round_next_i = len(history) + 1

    lines: list[str] = []
    if not history:
        lines.append(_format_main_answer_evaluation_markdown(result))
    else:
        lines.append(_format_single_follow_up_evaluation_markdown(history[-1]))
    lines.append(f"### 꼬리 질문 ({round_next_i}/3)\n\n{fu_q}")
    return "\n\n".join(lines)


def _format_final_session_markdown(result: dict[str, Any]) -> str:
    """세션 종료: 첫 답변 + 3회 꼬리 평가 + 학습 주제 + SQLite 저장 안내."""
    topics = result.get("tracked_learning_topic_labels") or []
    lines: list[str] = [_format_main_answer_evaluation_markdown(result)]
    for item in result.get("follow_up_history") or []:
        lines.append(_format_single_follow_up_evaluation_markdown(item))
    lines.append("### 종합")
    if topics:
        lines.append(f"**학습 포커스 주제**\n\n{', '.join(str(t) for t in topics)}")
    lines.append(
        "\n---\n"
        "*꼬리 질문 3회 답변과 평가가 끝났습니다. 복습 카드가 SQLite에 반영되었습니다. "
        "새로 시작하려면 사이드바의 **Start / Reset interview**를 누르세요.*"
    )
    return "\n\n".join(lines)


def _start_interview(topic: str, learner_identifier: str, profile: dict[str, Any] | None) -> None:
    graph = _ensure_graph()
    _init_message_list()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.config = config
    st.session_state.waiting_for_answer = False
    st.session_state.messages = []
    st.session_state.pop("resume_answer_pending", None)
    st.session_state.graph_answer_busy = False

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
                "content": _format_final_session_markdown(result),
            }
        )


def _snapshot_to_invoke_result(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Recreate LangGraph.invoke-style dict (__interrupt__) from persisted checkpoint snapshot."""
    snap = graph.get_state(config)
    merged: dict[str, Any] = dict(snap.values)
    if snap.interrupts:
        merged["__interrupt__"] = list(snap.interrupts)
    return merged


def _stream_answer_resume_then_snapshot(graph: Any, config: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Resume with user answer while streaming narration tokens from evaluator nodes."""
    cmd = Command(resume={"answer": user_text})

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        score_hint = st.empty()
        streamed = ""

        def consume_custom(chunk: dict[str, Any]) -> None:
            nonlocal streamed
            token = chunk.get(STREAM_EVALUATION_CHUNK_KEY)
            if isinstance(token, str) and token:
                streamed += token
                stream_placeholder.markdown(streamed + " ▍ ")
            phase = chunk.get(STREAM_EVALUATION_PHASE_KEY)
            if isinstance(phase, str) and phase == "scoring_started":
                score_hint.caption("루브릭 점수를 확정하는 중…")

        for mode, chunk in graph.stream(cmd, config=config, stream_mode=["updates", "custom"]):
            if mode == "custom" and isinstance(chunk, dict):
                consume_custom(chunk)

    return _snapshot_to_invoke_result(graph, config)


def _finalize_resume_invoke_result(result: dict[str, Any]) -> None:
    """Persist assistant markup like the legacy graph.invoke `_submit_answer` path."""
    intr = _interrupt_payload(result)
    if intr:
        st.session_state.waiting_for_answer = True
        intr_type = intr.get("type")
        if intr_type == "follow_up_answer":
            content = _format_follow_up_step_markdown(result, intr)
        else:
            content = (
                "### 계속\n\n"
                f"(알 수 없는 interrupt 유형: `{intr_type}`)"
            )
        st.session_state.messages.append({"role": "assistant", "content": content})
        return

    st.session_state.waiting_for_answer = False
    st.session_state.messages.append(
        {"role": "assistant", "content": _format_final_session_markdown(result)},
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
    graph_busy = bool(st.session_state.get("graph_answer_busy"))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    resume_text = st.session_state.pop("resume_answer_pending", None)
    if resume_text is not None:
        graph = _ensure_graph()
        config = st.session_state.get("config")
        st.session_state.graph_answer_busy = True
        try:
            if not config:
                st.error("먼저 사이드바에서 인터뷰를 시작하세요.")
            else:
                try:
                    result = _stream_answer_resume_then_snapshot(graph, config, resume_text)
                    _finalize_resume_invoke_result(result)
                except RuntimeError as exc:
                    st.error(f"실행 오류: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"예기치 않은 오류: {exc}")
        finally:
            st.session_state.graph_answer_busy = False
        st.rerun()

    chat_disabled = not waiting or graph_busy

    prompt = st.chat_input(
        "답변을 입력하세요…" if waiting else "사이드바에서 인터뷰를 시작하면 답변을 입력할 수 있습니다.",
        disabled=chat_disabled,
    )
    if prompt and waiting and not graph_busy:
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.resume_answer_pending = prompt.strip()
        st.session_state.graph_answer_busy = True
        st.rerun()


if __name__ == "__main__":
    main()
