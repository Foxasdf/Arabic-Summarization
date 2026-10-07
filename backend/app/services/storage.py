"""
SQLite persistence. Uses the stdlib sqlite3 module (via asyncio.to_thread) so it
works identically on Windows / Linux / macOS with no native extras.
Paths stored in the DB are never absolute, so the whole project folder
(app + models + data) can be copied to another machine as-is.
"""
import json
import logging
import asyncio
import sqlite3
from typing import List, Dict, Optional
from ..config import settings

logger = logging.getLogger(__name__)

_COLUMNS = {
    "id": "TEXT PRIMARY KEY",
    "filename": "TEXT",
    "upload_time": "DATETIME DEFAULT CURRENT_TIMESTAMP",
    "duration_seconds": "REAL",
    "transcript_json": "TEXT",
    "diarization_json": "TEXT",
    "speakers_json": "TEXT",
    "summary_text": "TEXT",
    "status": "TEXT",
}
_JSON_FIELDS = {"transcript_json": [], "diarization_json": [], "speakers_json": {}}


def _connect():
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _sync_init_db():
    conn = _connect()
    with conn:
        cols = ", ".join(f"{k} {v}" for k, v in _COLUMNS.items())
        conn.execute(f"CREATE TABLE IF NOT EXISTS meetings ({cols})")
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(meetings)")}
        for k, v in _COLUMNS.items():
            if k not in existing:  # lightweight migration for older databases
                conn.execute(f"ALTER TABLE meetings ADD COLUMN {k} {v.split(' DEFAULT')[0]}")
                logger.info(f"DB migration: added column {k}")
    conn.close()


def _row_to_dict(row) -> Dict:
    res = dict(row)
    for f, default in _JSON_FIELDS.items():
        try:
            res[f] = json.loads(res[f]) if res.get(f) else default
        except (TypeError, json.JSONDecodeError):
            res[f] = default
    return res


def _sync_save_meeting(data: dict) -> str:
    """Upsert: only provided (non-None) fields are written."""
    fields = {k: data[k] for k in _COLUMNS if k in data and k != "upload_time" and data[k] is not None}
    for f in _JSON_FIELDS:
        if f in fields:
            fields[f] = json.dumps(fields[f], ensure_ascii=False)
    keys = list(fields)
    placeholders = ", ".join("?" for _ in keys)
    updates = ", ".join(f"{k}=excluded.{k}" for k in keys if k != "id") or "id=id"
    conn = _connect()
    with conn:
        conn.execute(
            f"INSERT INTO meetings ({', '.join(keys)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            [fields[k] for k in keys],
        )
    conn.close()
    return data["id"]


def _sync_get_meeting(meeting_id: str) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def _sync_list_meetings() -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, filename, upload_time, duration_seconds, status FROM meetings ORDER BY upload_time DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _sync_delete_meeting(meeting_id: str):
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.close()


async def init_db():
    await asyncio.to_thread(_sync_init_db)


async def save_meeting(data: dict) -> str:
    return await asyncio.to_thread(_sync_save_meeting, data)


async def get_meeting(meeting_id: str) -> Optional[Dict]:
    return await asyncio.to_thread(_sync_get_meeting, meeting_id)


async def list_meetings() -> List[Dict]:
    return await asyncio.to_thread(_sync_list_meetings)


async def delete_meeting(meeting_id: str):
    await asyncio.to_thread(_sync_delete_meeting, meeting_id)
