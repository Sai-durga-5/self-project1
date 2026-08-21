import sqlite3
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./database/interview_log.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS override_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            step TEXT,
            candidate_name TEXT,
            ai_score INTEGER,
            ai_recommendation TEXT,
            human_decision TEXT,
            human_edited_score INTEGER,
            final_outcome TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_log(record: dict) -> None:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO override_log (
            timestamp, step, candidate_name, ai_score,
            ai_recommendation, human_decision, human_edited_score,
            final_outcome, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("timestamp", datetime.now().isoformat()),
        record.get("step", ""),
        record.get("candidate_name", ""),
        record.get("ai_score"),
        record.get("ai_recommendation", ""),
        record.get("human_decision", ""),
        record.get("human_edited_score"),
        record.get("final_outcome", ""),
        record.get("notes", ""),
    ))
    conn.commit()
    conn.close()


def get_all_logs() -> list[dict]:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM override_log ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_logs_by_candidate(name: str) -> list[dict]:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM override_log WHERE candidate_name = ? ORDER BY timestamp DESC",
        (name,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
