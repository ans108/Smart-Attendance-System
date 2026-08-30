"""
database.py — SQLite attendance store and CSV export.

v3 changes:
- Added embeddings table for ArcFace 512-d face embeddings.
- attendance.late_minutes column (minutes after work_start, 0 = on-time).
- attendance.status now uses 'on_time' | 'late' | 'very_late' (legacy 'Present' mapped on read).
- Analytics queries for admin dashboard (weekly chart, student analysis, summary).
- get_attendance() now returns status/late_minutes.
"""

import csv
import os
import sqlite3
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    NOT NULL,
                label   INTEGER NOT NULL UNIQUE,
                created TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id),
                name         TEXT    NOT NULL,
                date         TEXT    NOT NULL,
                time         TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'on_time',
                late_minutes INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_attendance_user_date
                ON attendance (user_id, date);

            CREATE INDEX IF NOT EXISTS ix_attendance_date_time
                ON attendance (date, time);

            CREATE INDEX IF NOT EXISTS ix_attendance_name
                ON attendance (name);

            CREATE TABLE IF NOT EXISTS embeddings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  INTEGER NOT NULL UNIQUE REFERENCES users(id),
                embedding BLOB NOT NULL,
                updated  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
        """)
        # Migrate older DBs: add columns if they don't exist
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add new columns to existing attendance table without data loss."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(attendance)").fetchall()}
    if "late_minutes" not in cols:
        conn.execute("ALTER TABLE attendance ADD COLUMN late_minutes INTEGER NOT NULL DEFAULT 0")
        print("[DB] Migrated: added late_minutes column.")
    # Create embeddings table if missing (safe no-op if already exists above)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL UNIQUE REFERENCES users(id),
            embedding BLOB NOT NULL,
            updated  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
    """)


# ── Users ─────────────────────────────────────────────────────────────────────

def add_user(name: str, label: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, label) VALUES (?, ?)", (name.strip(), label)
        )
        return cur.lastrowid


def get_all_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, label, created FROM users ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_by_label(label: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, label FROM users WHERE label = ?", (label,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, label FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_name(name: str) -> dict | None:
    """Find an existing user by display name without case sensitivity."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, label FROM users WHERE lower(name) = lower(?)",
            (name.strip(),),
        ).fetchone()
    return dict(row) if row else None


def next_label() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(label) as m FROM users").fetchone()
    return (row["m"] or 0) + 1


def delete_user(user_id: int) -> int | None:
    user = get_user_by_id(user_id)
    if user is None:
        return None
    label = user["label"]
    with get_connection() as conn:
        conn.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return label


# ── Embeddings ────────────────────────────────────────────────────────────────

def save_embedding(user_id: int, embedding_bytes: bytes) -> None:
    """Upsert a face embedding for a user."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO embeddings (user_id, embedding, updated)
               VALUES (?, ?, datetime('now','localtime'))
               ON CONFLICT(user_id) DO UPDATE SET
                   embedding = excluded.embedding,
                   updated   = excluded.updated""",
            (user_id, embedding_bytes),
        )


def get_all_embeddings() -> list[dict]:
    """
    Return all stored embeddings with user info.
    Each dict: { user_id, name, label, embedding (bytes) }
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT e.user_id, u.name, u.label, e.embedding
               FROM embeddings e JOIN users u ON e.user_id = u.id"""
        ).fetchall()
    return [dict(r) for r in rows]


def delete_embedding(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))


def embedding_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


# ── Attendance ────────────────────────────────────────────────────────────────

def mark_attendance(user_id: int, name: str,
                    status: str = "on_time",
                    late_minutes: int = 0) -> bool:
    """
    Mark attendance for today. Returns True if newly marked, False if already done.
    """
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO attendance (user_id, name, date, time, status, late_minutes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, name, today, now, status, late_minutes),
            )
        return True
    except sqlite3.IntegrityError:
        return False   # already marked today


def manual_mark_attendance(user_id: int, date_str: str, time_str: str,
                           status: str = "on_time",
                           late_minutes: int = 0) -> dict:
    """
    Insert or update attendance for a specific user/date.
    Returns {created: bool, record: dict}.
    """
    user = get_user_by_id(user_id)
    if user is None:
        raise ValueError("User not found")

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM attendance WHERE user_id = ? AND date = ?",
            (user_id, date_str),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE attendance
                   SET name = ?, time = ?, status = ?, late_minutes = ?,
                       created_at = datetime('now','localtime')
                   WHERE user_id = ? AND date = ?""",
                (user["name"], time_str, status, late_minutes, user_id, date_str),
            )
            created = False
        else:
            conn.execute(
                """INSERT INTO attendance (user_id, name, date, time, status, late_minutes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, user["name"], date_str, time_str, status, late_minutes),
            )
            created = True

        row = conn.execute(
            """SELECT a.*, u.label
               FROM attendance a JOIN users u ON a.user_id = u.id
               WHERE a.user_id = ? AND a.date = ?""",
            (user_id, date_str),
        ).fetchone()
    return {"created": created, "record": dict(row) if row else None}


def get_attendance(
    filter_date: str | None = None,
    date_from:   str | None = None,
    date_to:     str | None = None,
    name_query:  str | None = None,
) -> list[dict]:
    sql    = """SELECT a.*, u.label
                FROM attendance a JOIN users u ON a.user_id = u.id
                WHERE 1=1"""
    params: list = []

    if filter_date:
        sql += " AND a.date = ?"
        params.append(filter_date)
    else:
        if date_from:
            sql += " AND a.date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND a.date <= ?"
            params.append(date_to)

    if name_query:
        sql += " AND a.name LIKE ?"
        params.append(f"%{name_query}%")

    sql += " ORDER BY a.date DESC, a.time DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_today_stats() -> dict:
    today = date.today().isoformat()
    with get_connection() as conn:
        present = conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE date = ?", (today,)
        ).fetchone()["c"]
        late_count = conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE date = ? AND status IN ('late','very_late')",
            (today,)
        ).fetchone()["c"]
        total_users = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
        recent = conn.execute(
            """SELECT name, time, status, late_minutes
               FROM attendance WHERE date = ? ORDER BY time DESC LIMIT 5""",
            (today,),
        ).fetchall()
    return {
        "present":     present,
        "late":        late_count,
        "absent":      max(0, total_users - present),
        "total_users": total_users,
        "recent":      [dict(r) for r in recent],
        "date":        today,
    }


# ── Admin analytics ───────────────────────────────────────────────────────────

def get_weekly_chart(days: int = 7) -> list[dict]:
    """
    Returns last N days with on_time, late, very_late counts per day.
    Suitable for a stacked bar chart.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT date,
                      SUM(CASE WHEN status = 'on_time'   OR status = 'Present' THEN 1 ELSE 0 END) as on_time,
                      SUM(CASE WHEN status = 'late'      THEN 1 ELSE 0 END) as late,
                      SUM(CASE WHEN status = 'very_late' THEN 1 ELSE 0 END) as very_late
               FROM attendance
               GROUP BY date
               ORDER BY date DESC
               LIMIT ?""",
            (days,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_student_analysis() -> list[dict]:
    """
    Per-student attendance summary:
    name, total_days, on_time_days, late_days, very_late_days, avg_late_minutes, last_seen
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT
                 u.name,
                 COUNT(a.id)                       AS total_days,
                 SUM(CASE WHEN a.status IN ('on_time','Present') THEN 1 ELSE 0 END) AS on_time_days,
                 SUM(CASE WHEN a.status = 'late'      THEN 1 ELSE 0 END) AS late_days,
                 SUM(CASE WHEN a.status = 'very_late' THEN 1 ELSE 0 END) AS very_late_days,
                 ROUND(AVG(CASE WHEN a.late_minutes > 0 THEN a.late_minutes ELSE NULL END), 1) AS avg_late_minutes,
                 MAX(a.date) AS last_seen
               FROM users u
               LEFT JOIN attendance a ON u.id = a.user_id
               GROUP BY u.id, u.name
               ORDER BY u.name"""
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        total = d["total_days"] or 0
        on_t  = d["on_time_days"] or 0
        d["on_time_pct"] = round(100 * on_t / total, 1) if total > 0 else 0.0
        result.append(d)
    return result


def get_admin_overview() -> dict:
    """Summary stats for admin dashboard overview cards."""
    today = date.today().isoformat()
    with get_connection() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        today_present = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
        ).fetchone()[0]
        today_late = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date = ? AND status IN ('late','very_late')",
            (today,)
        ).fetchone()[0]
        # This-week attendance rate
        week_total = conn.execute(
            """SELECT COUNT(*) FROM attendance
               WHERE date >= date('now', '-6 days')"""
        ).fetchone()[0]
        week_possible = conn.execute(
            """SELECT COUNT(*) * 7 FROM users"""
        ).fetchone()[0] or 1
        week_rate = round(100 * week_total / week_possible, 1)
    return {
        "total_users":   total_users,
        "today_present": today_present,
        "today_late":    today_late,
        "week_rate":     week_rate,
        "today":         today,
    }


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_csv(path: str, filter_date: str | None = None) -> str:
    rows = get_attendance(filter_date=filter_date)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "date", "time", "status", "late_minutes", "created_at"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# Bootstrap on import
init_db()
