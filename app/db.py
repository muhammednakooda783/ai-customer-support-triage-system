from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta

_db_path: str | None = None


def init_db(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT,
                confidence REAL,
                suggested_reply TEXT,
                classifier_name TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                ok INTEGER NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_classifications_created_at "
            "ON classifications(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_classifications_category ON classifications(category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_classifications_classifier "
            "ON classifications(classifier_name)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_classifications_ok ON classifications(ok)")


def insert_classification(
    request_id: str,
    text: str,
    category: str | None,
    confidence: float | None,
    suggested_reply: str | None,
    classifier_name: str,
    latency_ms: int,
    ok: bool,
    error_message: str | None,
    created_at: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO classifications (
                request_id, text, category, confidence, suggested_reply,
                classifier_name, latency_ms, ok, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                text,
                category,
                confidence,
                suggested_reply,
                classifier_name,
                latency_ms,
                1 if ok else 0,
                error_message,
                created_at,
            ),
        )


def get_recent(
    limit: int = 20,
    category: str | None = None,
    classifier: str | None = None,
    status: str | None = None,
    q: str | None = None,
) -> list[dict]:
    safe_limit = max(1, min(100, int(limit)))
    query = """
        SELECT request_id, text, category, confidence, suggested_reply,
               classifier_name, latency_ms, ok, error_message, created_at
        FROM classifications
        WHERE 1=1
    """
    params: list[object] = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if classifier:
        query += " AND classifier_name = ?"
        params.append(classifier)
    if status == "ok":
        query += " AND ok = 1"
    elif status == "error":
        query += " AND ok = 0"
    if q:
        query += (
            " AND (text LIKE ? OR IFNULL(suggested_reply, '') LIKE ? "
            "OR IFNULL(error_message, '') LIKE ?)"
        )
        pattern = f"%{q}%"
        params.extend([pattern, pattern, pattern])
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(safe_limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_stats(window_minutes: int = 60) -> dict:
    safe_window = max(1, min(1440, int(window_minutes)))
    since = datetime.now(UTC) - timedelta(minutes=safe_window)
    since_iso = since.isoformat()

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT request_id, category, confidence, classifier_name, latency_ms, ok,
                   error_message, created_at
            FROM classifications
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (since_iso,),
        ).fetchall()

    total_requests = len(rows)
    ok_count = sum(1 for row in rows if bool(row["ok"]))
    ok_rate = (ok_count / total_requests) if total_requests else 0.0

    latency_values = [int(row["latency_ms"]) for row in rows]
    avg_latency_ms = (sum(latency_values) / len(latency_values)) if latency_values else 0.0

    confidence_values = [
        float(row["confidence"])
        for row in rows
        if row["confidence"] is not None and bool(row["ok"])
    ]
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

    category_counts = Counter(
        row["category"] for row in rows if row["category"] is not None and bool(row["ok"])
    )
    classifier_counts = Counter(row["classifier_name"] for row in rows)

    errors_last_10 = [
        {
            "request_id": str(row["request_id"]),
            "classifier_name": str(row["classifier_name"]),
            "error_message": str(row["error_message"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
        if not bool(row["ok"]) and row["error_message"]
    ][:10]

    return {
        "total_requests": total_requests,
        "ok_rate": round(ok_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "avg_confidence": round(avg_confidence, 4),
        "category_counts": dict(category_counts),
        "classifier_counts": dict(classifier_counts),
        "errors_last_10": errors_last_10,
        "last_updated_iso": datetime.now(UTC).isoformat(),
    }


def _connect() -> sqlite3.Connection:
    if not _db_path:
        raise RuntimeError("Database is not initialized. Call init_db(db_path) first.")
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "request_id": str(row["request_id"]),
        "text": str(row["text"]),
        "category": row["category"],
        "confidence": row["confidence"],
        "suggested_reply": row["suggested_reply"],
        "classifier_name": str(row["classifier_name"]),
        "latency_ms": int(row["latency_ms"]),
        "ok": bool(row["ok"]),
        "error_message": row["error_message"],
        "created_at": str(row["created_at"]),
    }
