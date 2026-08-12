from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .memory_store import default_bucket, utc_iso


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _json_loads(text: str | None, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _slugify(text: str) -> str:
    clean = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", str(text or ""))
    clean = clean.replace("_", " ").replace("-", " ")
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def _derive_series_key(source_file_name: str) -> str:
    stem = Path(str(source_file_name or "")).stem
    clean = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", stem)
    clean = re.sub(r"\b\d{1,4}\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return _slugify(clean or stem)


def _derive_issue_key(source_file_name: str) -> str:
    stem = Path(str(source_file_name or "")).stem
    clean = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", stem)
    clean = re.sub(r"\s+", " ", clean).strip()
    return _slugify(clean or stem)


class SQLiteSharedMemoryStore:
    bucket_to_memory_type = {
        "glossary": "glossary",
        "translation_memory": "translation",
        "style_memory": "style",
        "project_memory": "project",
        "experience_memory": "experience",
    }
    supported_buckets = tuple(bucket_to_memory_type.keys())

    def __init__(self, db_path: Optional[str] = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "data" / "memory" / "shared_memory.db"
        env_path = str(os.getenv("MANGA_AGENT_SHARED_MEMORY_DB") or "").strip()
        chosen_path = db_path or env_path or str(default_path)
        self.db_path = Path(chosen_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_uuid TEXT NOT NULL UNIQUE,
                    memory_type TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_value TEXT,
                    source_text TEXT,
                    target_text TEXT,
                    note TEXT,
                    source_kind TEXT NOT NULL,
                    glossary_type TEXT,
                    recommended_action TEXT,
                    stage TEXT,
                    page_type TEXT,
                    language_pair TEXT,
                    series_key TEXT,
                    issue_key TEXT,
                    source_file_name TEXT,
                    locked INTEGER NOT NULL DEFAULT 0,
                    confidence REAL,
                    repeat_count INTEGER NOT NULL DEFAULT 1,
                    skill_params_json TEXT,
                    tags_json TEXT,
                    extra_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memory_type_scope
                    ON memory_entries(memory_type, scope_type, scope_value);
                CREATE INDEX IF NOT EXISTS idx_memory_type_series
                    ON memory_entries(memory_type, series_key);
                CREATE INDEX IF NOT EXISTS idx_memory_language_pair
                    ON memory_entries(memory_type, language_pair);
                CREATE INDEX IF NOT EXISTS idx_memory_stage_page_type
                    ON memory_entries(stage, page_type);
                CREATE INDEX IF NOT EXISTS idx_memory_updated_at
                    ON memory_entries(updated_at);
                CREATE TABLE IF NOT EXISTS memory_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_uuid TEXT NOT NULL,
                    job_id TEXT,
                    page_index INTEGER,
                    route TEXT NOT NULL,
                    reason TEXT,
                    used_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _job_context(self, job) -> Dict[str, Any]:
        source_file_name = str(getattr(job, "source_file_name", "") or "").strip()
        source_language = str((getattr(job, "user_preferences", {}) or {}).get("source_language") or "en").strip().lower()
        target_language = str(getattr(job, "target_language", "") or "").strip().lower()
        language_pair = f"{source_language}->{target_language}" if source_language and target_language else None
        return {
            "job_id": str(getattr(job, "job_id", "") or "").strip(),
            "source_file_name": source_file_name,
            "series_key": _derive_series_key(source_file_name),
            "issue_key": _derive_issue_key(source_file_name),
            "language_pair": language_pair,
        }

    def _infer_scope(self, bucket: str, job, entry: Dict[str, Any], context: Dict[str, Any]) -> tuple[str, Optional[str]]:
        scope_type = str(entry.get("scope_type") or "").strip().lower()
        scope_value = str(entry.get("scope_value") or "").strip() or None
        if scope_type:
            return scope_type, scope_value

        if bucket == "experience_memory":
            return "global", None

        if bucket in {"glossary", "translation_memory", "style_memory", "project_memory"}:
            if context.get("series_key"):
                return "series", context["series_key"]
        return "global", None

    def _prepare_entry(self, bucket: str, job, raw: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_iso()
        context = self._job_context(job)
        entry = dict(raw or {})
        scope_type, scope_value = self._infer_scope(bucket, job, entry, context)
        memory_type = self.bucket_to_memory_type[bucket]
        skill_params = entry.get("skill_params") if isinstance(entry.get("skill_params"), dict) else {}
        tags = [str(x).strip() for x in (entry.get("tags") or []) if str(x).strip()]
        situation = entry.get("situation") if isinstance(entry.get("situation"), dict) else {}
        note = str(entry.get("note") or entry.get("why") or "").strip()
        glossary_type = str(entry.get("type") or entry.get("glossary_type") or "").strip().lower() or None
        if glossary_type not in {"term", "quote", "name", "phrase"}:
            glossary_type = None

        try:
            confidence = float(entry.get("confidence")) if entry.get("confidence") is not None else None
        except Exception:
            confidence = None

        return {
            "entry_uuid": str(entry.get("entry_id") or entry.get("entry_uuid") or uuid.uuid4()),
            "memory_type": memory_type,
            "scope_type": scope_type or "global",
            "scope_value": scope_value,
            "source_text": str(entry.get("src_text") or "").strip() or None,
            "target_text": str(entry.get("tgt_text") or "").strip() or None,
            "note": note or str(entry.get("tgt_text") or "").strip() or None,
            "source_kind": str(entry.get("source") or "system").strip() or "system",
            "glossary_type": glossary_type,
            "recommended_action": str(entry.get("recommended_action") or "").strip() or None,
            "stage": str(entry.get("stage") or situation.get("stage") or "").strip().lower() or None,
            "page_type": str(entry.get("page_type") or situation.get("page_type") or "").strip().lower() or None,
            "language_pair": context.get("language_pair"),
            "series_key": context.get("series_key") or None,
            "issue_key": context.get("issue_key") or None,
            "source_file_name": context.get("source_file_name") or None,
            "locked": 1 if bool(entry.get("locked")) else 0,
            "confidence": confidence,
            "repeat_count": max(1, int(entry.get("repeat_count") or 1)),
            "skill_params_json": _json_dumps(skill_params),
            "tags_json": _json_dumps(tags),
            "extra_json": _json_dumps(entry),
            "created_at": str(entry.get("created_at") or entry.get("first_seen_at") or entry.get("updated_at") or now),
            "updated_at": str(entry.get("updated_at") or entry.get("last_seen_at") or now),
        }

    def _find_existing_row(self, conn: sqlite3.Connection, prepared: Dict[str, Any]) -> Optional[sqlite3.Row]:
        return conn.execute(
            """
            SELECT *
            FROM memory_entries
            WHERE memory_type = ?
              AND scope_type = ?
              AND IFNULL(scope_value, '') = IFNULL(?, '')
              AND IFNULL(source_text, '') = IFNULL(?, '')
              AND IFNULL(target_text, '') = IFNULL(?, '')
              AND IFNULL(recommended_action, '') = IFNULL(?, '')
            LIMIT 1
            """,
            (
                prepared["memory_type"],
                prepared["scope_type"],
                prepared["scope_value"],
                prepared["source_text"],
                prepared["target_text"],
                prepared["recommended_action"],
            ),
        ).fetchone()

    def _merge_payload(self, existing: sqlite3.Row, prepared: Dict[str, Any]) -> Dict[str, Any]:
        merged_tags = sorted(
            {
                str(x).strip()
                for x in _json_loads(existing["tags_json"], []) + _json_loads(prepared["tags_json"], [])
                if str(x).strip()
            }
        )
        merged_skill_params = dict(_json_loads(existing["skill_params_json"], {}))
        merged_skill_params.update(_json_loads(prepared["skill_params_json"], {}))
        existing_extra = _json_loads(existing["extra_json"], {})
        prepared_extra = _json_loads(prepared["extra_json"], {})
        merged_extra = {**existing_extra, **prepared_extra}

        old_conf = existing["confidence"]
        new_conf = prepared["confidence"]
        try:
            confidence = max(x for x in [old_conf, new_conf] if x is not None)
        except Exception:
            confidence = new_conf if new_conf is not None else old_conf

        return {
            "entry_uuid": str(existing["entry_uuid"] or prepared["entry_uuid"]),
            "note": prepared["note"] or existing["note"],
            "source_kind": prepared["source_kind"] or existing["source_kind"],
            "glossary_type": prepared["glossary_type"] or existing["glossary_type"],
            "stage": prepared["stage"] or existing["stage"],
            "page_type": prepared["page_type"] or existing["page_type"],
            "language_pair": prepared["language_pair"] or existing["language_pair"],
            "series_key": prepared["series_key"] or existing["series_key"],
            "issue_key": prepared["issue_key"] or existing["issue_key"],
            "source_file_name": prepared["source_file_name"] or existing["source_file_name"],
            "locked": 1 if (int(existing["locked"] or 0) or int(prepared["locked"] or 0)) else 0,
            "confidence": confidence,
            "repeat_count": int(existing["repeat_count"] or 1) + max(1, int(prepared["repeat_count"] or 1)),
            "skill_params_json": _json_dumps(merged_skill_params),
            "tags_json": _json_dumps(merged_tags),
            "extra_json": _json_dumps(merged_extra),
            "updated_at": utc_iso(),
        }

    def upsert_entries(self, job, bucket: str, entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if bucket not in self.supported_buckets:
            return []

        normalized_entries = [x for x in entries if isinstance(x, dict)]
        if not normalized_entries:
            return []

        written: List[Dict[str, Any]] = []
        with self._connect() as conn:
            for raw in normalized_entries:
                prepared = self._prepare_entry(bucket, job, raw)
                existing = self._find_existing_row(conn, prepared)
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO memory_entries (
                            entry_uuid, memory_type, scope_type, scope_value, source_text, target_text,
                            note, source_kind, glossary_type, recommended_action, stage, page_type,
                            language_pair, series_key, issue_key, source_file_name, locked, confidence,
                            repeat_count, skill_params_json, tags_json, extra_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prepared["entry_uuid"],
                            prepared["memory_type"],
                            prepared["scope_type"],
                            prepared["scope_value"],
                            prepared["source_text"],
                            prepared["target_text"],
                            prepared["note"],
                            prepared["source_kind"],
                            prepared["glossary_type"],
                            prepared["recommended_action"],
                            prepared["stage"],
                            prepared["page_type"],
                            prepared["language_pair"],
                            prepared["series_key"],
                            prepared["issue_key"],
                            prepared["source_file_name"],
                            prepared["locked"],
                            prepared["confidence"],
                            prepared["repeat_count"],
                            prepared["skill_params_json"],
                            prepared["tags_json"],
                            prepared["extra_json"],
                            prepared["created_at"],
                            prepared["updated_at"],
                        ),
                    )
                else:
                    merged = self._merge_payload(existing, prepared)
                    conn.execute(
                        """
                        UPDATE memory_entries
                        SET entry_uuid = ?, note = ?, source_kind = ?, glossary_type = ?, recommended_action = ?,
                            stage = ?, page_type = ?, language_pair = ?, series_key = ?, issue_key = ?,
                            source_file_name = ?, locked = ?, confidence = ?, repeat_count = ?,
                            skill_params_json = ?, tags_json = ?, extra_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            merged["entry_uuid"],
                            merged["note"],
                            merged["source_kind"],
                            merged["glossary_type"],
                            prepared["recommended_action"],
                            merged["stage"],
                            merged["page_type"],
                            merged["language_pair"],
                            merged["series_key"],
                            merged["issue_key"],
                            merged["source_file_name"],
                            merged["locked"],
                            merged["confidence"],
                            merged["repeat_count"],
                            merged["skill_params_json"],
                            merged["tags_json"],
                            merged["extra_json"],
                            merged["updated_at"],
                            int(existing["id"]),
                        ),
                    )
                written.append(prepared)
            conn.commit()
        return written

    def append_experience_event(
        self,
        job,
        *,
        kind: str,
        note: str,
        situation: Optional[Dict[str, Any]] = None,
        recommended_action: Optional[str] = None,
        skill_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        raw = {
            "entry_id": str(uuid.uuid4()),
            "kind": str(kind or "runtime_observation"),
            "src_text": _json_dumps(situation or {}),
            "tgt_text": str(note or "").strip(),
            "note": str(note or "").strip(),
            "scope": "project",
            "source": "runtime_observation",
            "updated_at": utc_iso(),
            "situation": dict(situation or {}),
            "recommended_action": str(recommended_action or "").strip() or None,
            "skill_params": dict(skill_params or {}),
            "tags": [str(x).strip() for x in (tags or []) if str(x).strip()],
            "confidence": confidence,
            "repeat_count": 1,
        }
        self.upsert_entries(job, "experience_memory", [raw])
        return raw

    def _scope_candidates(self, job) -> List[tuple[str, Optional[str]]]:
        context = self._job_context(job)
        candidates: List[tuple[str, Optional[str]]] = [("global", None)]
        if context.get("series_key"):
            candidates.insert(0, ("series", context["series_key"]))
        if context.get("issue_key"):
            candidates.insert(0, ("issue", context["issue_key"]))
        return candidates

    def _row_to_entry(self, row: sqlite3.Row) -> Dict[str, Any]:
        extra = _json_loads(row["extra_json"], {})
        entry = dict(extra if isinstance(extra, dict) else {})
        entry.setdefault("entry_id", row["entry_uuid"])
        entry["entry_uuid"] = row["entry_uuid"]
        entry["src_text"] = row["source_text"] or entry.get("src_text")
        entry["tgt_text"] = row["target_text"] or entry.get("tgt_text")
        entry["note"] = row["note"] or entry.get("note")
        entry["source"] = row["source_kind"] or entry.get("source")
        entry["locked"] = bool(row["locked"])
        entry["confidence"] = row["confidence"]
        entry["updated_at"] = row["updated_at"]
        entry["scope"] = row["scope_type"] or entry.get("scope")
        entry["scope_type"] = row["scope_type"]
        entry["scope_value"] = row["scope_value"]
        entry["series_key"] = row["series_key"]
        entry["issue_key"] = row["issue_key"]
        entry["source_file_name"] = row["source_file_name"]
        entry["language_pair"] = row["language_pair"]
        entry["tags"] = _json_loads(row["tags_json"], [])
        skill_params = _json_loads(row["skill_params_json"], {})
        if skill_params:
            entry["skill_params"] = skill_params
        if row["memory_type"] == "experience":
            entry["kind"] = entry.get("kind") or "runtime_observation"
            entry["recommended_action"] = row["recommended_action"] or entry.get("recommended_action")
            entry["repeat_count"] = int(row["repeat_count"] or 1)
        if row["glossary_type"]:
            entry.setdefault("type", row["glossary_type"])
            entry["glossary_type"] = row["glossary_type"]
        entry["_storage_backend"] = "sqlite_shared"
        return entry

    def log_usage(
        self,
        *,
        entries: Iterable[Dict[str, Any]],
        route: str,
        job_id: Optional[str] = None,
        page_index: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> int:
        seen: set[str] = set()
        payload: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("_storage_backend") or "").strip() != "sqlite_shared":
                continue
            entry_uuid = str(entry.get("entry_uuid") or entry.get("entry_id") or "").strip()
            if not entry_uuid or entry_uuid in seen:
                continue
            seen.add(entry_uuid)
            payload.append(entry_uuid)

        if not payload:
            return 0

        now = utc_iso()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memory_usage_log(entry_uuid, job_id, page_index, route, reason, used_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(entry_uuid, job_id, page_index, route, reason, now) for entry_uuid in payload],
            )
            conn.executemany(
                """
                UPDATE memory_entries
                SET use_count = use_count + 1,
                    last_used_at = ?
                WHERE entry_uuid = ?
                """,
                [(now, entry_uuid) for entry_uuid in payload],
            )
            conn.commit()
        return len(payload)

    def snapshot_for_job(self, job, *, buckets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        wanted = [b for b in (buckets or self.supported_buckets) if b in self.supported_buckets]
        snap = {bucket: default_bucket(bucket) for bucket in wanted}
        if not wanted:
            return snap

        memory_types = [self.bucket_to_memory_type[bucket] for bucket in wanted]
        scope_candidates = self._scope_candidates(job)
        where_scope = " OR ".join("(scope_type = ? AND IFNULL(scope_value, '') = IFNULL(?, ''))" for _ in scope_candidates)
        params: List[Any] = memory_types + [item for pair in scope_candidates for item in pair]

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_entries
                WHERE memory_type IN ({",".join("?" for _ in memory_types)})
                  AND ({where_scope})
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchall()

        by_bucket = {bucket: [] for bucket in wanted}
        reverse_map = {v: k for k, v in self.bucket_to_memory_type.items()}
        for row in rows:
            bucket = reverse_map.get(str(row["memory_type"]))
            if bucket not in by_bucket:
                continue
            by_bucket[bucket].append(self._row_to_entry(row))

        for bucket in wanted:
            snap[bucket]["entries"] = by_bucket[bucket]
        return snap
