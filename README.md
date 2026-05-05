# Dev Coach

LangGraph 기반 Stateful 개발자 면접 코치

단순히 질문을 던지는 AI가 아니라,
사용자의 약점을 기억하고 반복 학습시키는 면접 에이전트


## 왜 만들었나?

개발자 면접 준비 서비스는 많습니다.

하지만 대부분은 아래 흐름에서 끝납니다.

질문 → 답변 → 끝

이 구조의 문제:

* 내가 반복적으로 틀리는 개념을 기억하지 못함
* 꼬리 질문 연습이 부족함
* 이전 면접 결과가 다음 학습에 반영되지 않음
* 장기적인 성장 추적이 어려움


## 우리가 만들고 싶은 경험

질문
→ 답변
→ 실시간 평가
→ 꼬리 질문
→ 약점 분석
→ 장기 저장
→ 복습 스케줄링
→ 다음 면접에 반영

사용자는 면접을 볼수록 더 똑똑한 코치를 갖게 됩니다.


## 핵심 가치

### 1. Adaptive Interview

사용자 수준에 따라 질문 난이도 조절

입력 정보:

* 경력
* 지원 회사
* 직무
* 이전 면접 결과

예시:

React란?
→ Reconciliation 설명
→ Fiber 구조 설명
→ Concurrent Rendering 설명


### 2. Dynamic Follow-up Questions

답변 품질을 평가하고 꼬리 질문 생성

평가 기준:

* 기술 정확성
* 깊이
* 실무 경험
* 전달력

예시:

"React Fiber 설명이 부족함"
→ Fiber Scheduler 꼬리 질문 생성


### 3. Weakness Memory

반복적으로 약한 개념 저장

예:

* Browser Rendering
* Network Caching
* System Design
* Database Index


### 4. Spaced Repetition

약한 질문 자동 재출제

Day 1
Day 3
Day 7
Day 14

실제 면접 전에 취약 개념을 반복 학습합니다.


## Why LangGraph?

일반 LLM 챗봇으로 구현하면:

질문 → 답변 → 응답

여기서 끝납니다.

LangGraph는 다음을 자연스럽게 구현할 수 있습니다:

* 상태 유지 (State Management)
* 조건 분기 (Conditional Routing)
* Memory Retrieval
* 반복 학습 루프
* Human-in-the-loop 확장


## Graph Flow

```
START
 ↓
Profile Loader
 ↓
Memory Retrieval
 ↓
Session Planner
 ↓
Question Generator
 ↓
User Answer
 ↓
Answer Evaluator
 ↓
Weakness Analyzer
 ↓
Decision Router
 ├── Follow-up Question
 ├── Next Question
 └── End Session
 ↓
Memory Update
 ↓
Review Scheduler
 ↓
END
```


## Memory Architecture

### Short-term Memory

LangGraph Checkpointer 활용

저장 정보:

* 현재 질문
* 최근 답변
* 꼬리 질문 흐름


### Long-term Memory

Supabase DB 저장

* 강점
* 약점
* 면접 히스토리
* 복습 일정


### Retrieval Layer

다음 세션 시작 시:

* overdue 질문 조회
* 약한 주제 조회
* 최근 실패 기록 조회


## User Flow

### 첫 사용

회원가입
→ 이력 입력
→ 목표 회사 설정
→ 첫 모의 면접 진행


### 반복 사용

복습 알림
→ 취약 질문 재도전
→ 난이도 상승
→ 실력 향상


## Tech Stack

* LangGraph
* OpenAI / Anthropic
* Supabase
* pgvector
* React
* TypeScript


## MVP Roadmap

### v1

* 면접 질문 생성
* 답변 평가
* 꼬리 질문 생성
* 약점 저장


### v2

* Spaced repetition
* 회사별 면접 모드
* 음성 인터뷰


### v3

* 실시간 코딩 인터뷰
* 시스템 디자인 인터뷰
* AI 면접 리포트 자동 생성


## 차별점

기존 서비스:

매번 새로운 면접

Dev Coach:

누적 학습
→ 약점 기억
→ 반복 학습
→ 성장 추적


## 한 줄 요약

“어제 틀린 걸 기억하고, 다음 면접에서 다시 물어보는 AI 면접 코치”