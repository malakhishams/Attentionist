"""
monitoring.py

Lightweight SQLite logging for Attentionist interactions and feedback.
Used by app.py to log each Q&A and by pages/dashboard.py to visualize it.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "processed" / "monitoring.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_json TEXT,
            num_sources INTEGER,
            answer_length INTEGER,
            feedback TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_interaction(question, answer, sources):
    """Logs a new Q&A interaction. Returns the row id for later feedback updates."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        INSERT INTO interactions (timestamp, question, answer, sources_json, num_sources, answer_length)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            question,
            answer,
            json.dumps([s.get("filename", "unknown") for s in sources]) if sources else "[]",
            len(sources) if sources else 0,
            len(answer),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def update_feedback(interaction_id, feedback):
    """feedback should be 'up' or 'down'."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE interactions SET feedback = ? WHERE id = ?",
        (feedback, interaction_id),
    )
    conn.commit()
    conn.close()


def load_all_interactions():
    """Returns all interactions as a list of dicts, for the dashboard."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM interactions ORDER BY timestamp").fetchall()
    conn.close()
    return [dict(row) for row in rows]