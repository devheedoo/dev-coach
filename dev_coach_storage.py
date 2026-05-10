"""SQLite persistence for Dev Coach review cards (one learner per client-side identifier)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


_DEFAULT_DB_NAME = "dev_coach.sqlite"


def sqlite_database_path() -> Path:
    raw = os.getenv("DEV_COACH_SQLITE_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd() / _DEFAULT_DB_NAME


def _connect() -> sqlite3.Connection:
    path = sqlite_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_cards (
                learner_identifier TEXT NOT NULL,
                topic_label TEXT NOT NULL,
                memory_easiness_factor REAL NOT NULL,
                successful_repetition_count INTEGER NOT NULL,
                interval_until_next_review_days REAL NOT NULL,
                next_review_unix_timestamp REAL NOT NULL,
                last_understanding_confidence_score_1_to_10 INTEGER,
                PRIMARY KEY (learner_identifier, topic_label)
            )
            """
        )
        conn.commit()


def load_review_cards(learner_identifier: str) -> list[dict[str, Any]]:
    init_database()
    learner_identifier = learner_identifier.strip() or "default"
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT topic_label,
                   memory_easiness_factor,
                   successful_repetition_count,
                   interval_until_next_review_days,
                   next_review_unix_timestamp,
                   last_understanding_confidence_score_1_to_10
            FROM review_cards
            WHERE learner_identifier = ?
            ORDER BY topic_label ASC
            """,
            (learner_identifier,),
        ).fetchall()
    cards: list[dict[str, Any]] = []
    for row in rows:
        score = row["last_understanding_confidence_score_1_to_10"]
        cards.append(
            {
                "topic_label": str(row["topic_label"]),
                "memory_easiness_factor": float(row["memory_easiness_factor"]),
                "successful_repetition_count": int(row["successful_repetition_count"]),
                "interval_until_next_review_days": float(row["interval_until_next_review_days"]),
                "next_review_unix_timestamp": float(row["next_review_unix_timestamp"]),
                "last_understanding_confidence_score_1_to_10": int(score)
                if score is not None
                else None,
            }
        )
    return cards


def replace_all_review_cards(learner_identifier: str, cards: list[dict[str, Any]]) -> None:
    """Replace every stored card for the learner with the provided snapshot."""
    init_database()
    learner_identifier = learner_identifier.strip() or "default"
    normalized: list[tuple[Any, ...]] = []
    for card in cards:
        topic = str(card.get("topic_label", "")).strip()
        if not topic:
            continue
        score = card.get("last_understanding_confidence_score_1_to_10")
        normalized.append(
            (
                learner_identifier,
                topic,
                float(card.get("memory_easiness_factor", 2.5)),
                int(card.get("successful_repetition_count", 0)),
                float(card.get("interval_until_next_review_days", 1.0)),
                float(card.get("next_review_unix_timestamp", 0.0)),
                int(score) if isinstance(score, int) else None,
            )
        )

    with _connect() as conn:
        conn.execute("DELETE FROM review_cards WHERE learner_identifier = ?", (learner_identifier,))
        conn.executemany(
            """
            INSERT INTO review_cards (
                learner_identifier,
                topic_label,
                memory_easiness_factor,
                successful_repetition_count,
                interval_until_next_review_days,
                next_review_unix_timestamp,
                last_understanding_confidence_score_1_to_10
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            normalized,
        )
        conn.commit()
