# Dev Coach

LangGraph 기반 개발자 면접 코치입니다. 사용자의 답변을 한국어로 평가하고, 꼬리 질문을 3회까지 이어 가며, 학습 포커스 주제를 SQLite 복습 카드로 저장합니다.

질문을 한 번 던지고 끝내는 챗봇이 아니라, 이전 답변에서 드러난 학습 주제를 다음 세션의 복습 질문으로 다시 가져오는 흐름을 목표로 합니다.

## 주요 기능

* 주제와 선택 프로필 기반 기술 면접 질문 생성
* `interrupt()` 기반 Human-in-the-loop 답변 수집
* 평가 서술 스트리밍과 구조화 루브릭 점수 분리
* 기술적 정확성, 설명 깊이, 실무 근거, 명확성·구조를 각 1~10점으로 평가
* 모범 답안 예시와 구체적인 피드백 제공
* 원본 답변 이후 최대 3회의 꼬리 질문 생성·평가
* 질문·답변·평가에서 학습 포커스 주제를 추출해 복습 카드로 저장
* SQLite 기반 학습자별 장기 메모리와 spaced repetition 스케줄링
* 예약 복습 질문 생성 시 OpenAI hosted web search를 선택적으로 활용
* Streamlit 채팅 UI와 LangGraph Studio/API 실행 지원

## Tech Stack

* Python 3.13+
* LangGraph
* LangChain OpenAI
* OpenAI Responses API
* Streamlit
* SQLite
* `uv`

## 실행 준비

의존성을 설치합니다.

```bash
uv sync
```

`.env.example`을 참고해 프로젝트 루트에 `.env`를 만들거나 shell 환경 변수로 설정합니다.

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_RESEARCH_MODEL=gpt-4o-mini
LANGSMITH_API_KEY=
LANGCHAIN_PROJECT=dev-coach
DEV_COACH_SQLITE_PATH=dev_coach.sqlite
DEV_COACH_LEARNER_ID=default
```

필수 환경 변수는 `OPENAI_API_KEY`입니다. `OPENAI_MODEL`의 기본값은 `gpt-4o-mini`이고, `OPENAI_RESEARCH_MODEL`은 예약 복습 질문의 웹 검색 보강에 사용됩니다. 웹 검색을 끄려면 `DEV_COACH_DISABLE_WEB_SEARCH=1`을 설정하세요.

SQLite 파일 경로는 `DEV_COACH_SQLITE_PATH`로 바꿀 수 있습니다. 설정하지 않으면 현재 작업 디렉터리의 `dev_coach.sqlite`를 사용합니다.

## Streamlit UI

브라우저에서 채팅 형태로 면접을 진행하려면 다음 명령을 실행합니다.

```bash
uv run streamlit run streamlit_app.py
```

사이드바에서 주제, 학습자 ID, 프로필 정보를 입력하고 `Start / Reset interview`를 누르면 새 인터뷰 thread가 시작됩니다. 답변을 입력하면 LangGraph가 resume되며, 평가 서술은 스트리밍으로 표시되고 루브릭 점수는 구조화 출력으로 확정됩니다.

## LangGraph Studio/API

`langgraph.json`에 정의된 그래프 ID는 `dev_coach`이고 entrypoint는 `dev_coach.py`의 `graph`입니다.

```bash
uv run langgraph dev
# 브라우저 자동 실행을 원하지 않으면:
# uv run langgraph dev --no-browser
```

그래프 입력 예시는 다음과 같습니다.

```json
{
  "topic": "React performance",
  "learner_identifier": "streamlit_user",
  "prefer_new_topic": true,
  "profile": {
    "years_experience": 5,
    "role": "Frontend Engineer",
    "level": "Senior",
    "target_company": "",
    "notes": ""
  }
}
```

`collect_answer`와 `collect_follow_up_answer` 노드는 `interrupt()`로 멈춥니다. Studio/API에서 재개할 때는 답변 문자열 또는 `{"answer": "..."}` 형태의 JSON을 전달하면 됩니다.

## Graph Flow

```text
START
  ↓
load_review_cards_from_sqlite
  ↓
pick_review_or_new_topic
  ├─ scheduled_review → generate_scheduled_review_question
  └─ new_topic        → generate_question
  ↓
collect_answer
  ↓
evaluate_answer
  ├─ update_review_schedule_after_evaluation
  └─ analyze_learning_focus
  ↓
merge_main_learning_topics
  ↓
generate_follow_up
  ↓
collect_follow_up_answer
  ↓
evaluate_follow_up_answer
  ↓
analyze_follow_up_learning_focus
  ↓
merge_follow_up_learning_topics
  ├─ more_follow_up → generate_follow_up
  └─ persist        → persist_review_cards_sqlite
  ↓
END
```

## Memory Architecture

### Short-Term Memory

LangGraph checkpointer가 한 인터뷰 thread의 진행 상태를 유지합니다. Streamlit UI에서는 `MemorySaver`를 사용해 현재 질문, 답변 대기 상태, 꼬리 질문 이력, 중간 평가 결과를 이어 갑니다.

### Long-Term Memory

`dev_coach_storage.py`는 SQLite의 `review_cards` 테이블에 학습자별 복습 카드를 저장합니다.

저장되는 주요 필드는 다음과 같습니다.

* `learner_identifier`
* `topic_label`
* `memory_easiness_factor`
* `successful_repetition_count`
* `interval_until_next_review_days`
* `next_review_unix_timestamp`
* `last_understanding_confidence_score_1_to_10`

### Spaced Repetition

복습 카드의 `next_review_unix_timestamp`가 현재 시각보다 이르면 새 주제보다 예약 복습 질문을 우선 생성합니다. 평가 평균은 0~5 품질 점수로 변환되고, 성공 횟수와 easiness factor에 따라 다음 복습 간격이 갱신됩니다.

## 노트북 스모크 테스트

`main.ipynb`에서는 로컬에서 그래프를 기본 시나리오로 한 바퀴 실행해 볼 수 있습니다.

흐름은 새 질문 생성, `interrupt()` 대기, `Command(resume=...)` 재개, 답변 평가, 학습 포커스 추출, SQLite 복습 카드 저장, 꼬리 질문 생성 순서입니다.

노트북처럼 직접 그래프를 실행하는 환경에서는 `build_graph(checkpointer=MemorySaver())`처럼 checkpointer를 명시적으로 전달해야 Human-in-the-loop 이후 상태를 이어 갈 수 있습니다.

## 현재 구현 범위

현재 저장소는 Python 단일 앱 중심의 MVP입니다. Supabase, pgvector, React/TypeScript UI, 음성 인터뷰, 코딩 인터뷰 기능은 아직 구현되어 있지 않습니다.

## 한 줄 요약

어제 부족했던 개념을 기억하고, 다음 면접에서 다시 물어보는 AI 면접 코치.