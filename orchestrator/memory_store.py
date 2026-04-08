from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_bucket(name: str) -> Dict[str, Any]:
    return {"version": "0.1.0", "bucket": name, "entries": [], "updated_at": utc_iso()}


class LocalMemoryStore:
    """Local JSON-backed memory buckets for one job."""

    bucket_files = {
        "glossary": "glossary.json",
        "translation_memory": "translation_memory.json",
        "style_memory": "style_memory.json",
        "project_memory": "project_memory.json",
        "brain_memory": "brain_memory.json",
    }
    rag_bundle_buckets = ("glossary", "translation_memory", "style_memory", "project_memory")

    def resolve_job_dir(self, job) -> Path:
        pref = str((job.user_preferences or {}).get("job_dir") or "").strip()
        if pref:
            return Path(pref)

        for page in getattr(job, "pages", {}).values():
            image_ref = str(getattr(page, "image_ref", "") or "").strip()
            if not image_ref:
                continue
            p = Path(image_ref)
            if not p.is_absolute() and p.parent == Path("."):
                continue
            if p.parent.name in {"pages", "erased", "rendered", "masks"}:
                return p.parent.parent
            if p.parent != Path("."):
                return p.parent

        return Path("data/jobs") / str(job.job_id)

    def memory_dir(self, job) -> Path:
        return self.resolve_job_dir(job) / "memory"

    def ensure_initialized(self, job) -> None:
        mem_dir = self.memory_dir(job)
        mem_dir.mkdir(parents=True, exist_ok=True)
        for bucket in self.bucket_files:
            path = mem_dir / self.bucket_files[bucket]
            if not path.exists():
                path.write_text(json.dumps(default_bucket(bucket), ensure_ascii=False, indent=2), encoding="utf-8")

        self._sync_job_glossary_noinit(job)

    def load_bucket(self, job, bucket: str) -> Dict[str, Any]:
        self.ensure_initialized(job)
        path = self.memory_dir(job) / self.bucket_files[bucket]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = default_bucket(bucket)
        if not isinstance(data, dict):
            data = default_bucket(bucket)
        if not isinstance(data.get("entries"), list):
            data["entries"] = []
        data.setdefault("version", "0.1.0")
        data.setdefault("bucket", bucket)
        data.setdefault("updated_at", utc_iso())
        return data

    def save_bucket(self, job, bucket: str, data: Dict[str, Any]) -> None:
        self.ensure_initialized(job)
        data2 = dict(data or {})
        data2["version"] = "0.1.0"
        data2["bucket"] = bucket
        if not isinstance(data2.get("entries"), list):
            data2["entries"] = []
        data2["updated_at"] = utc_iso()
        path = self.memory_dir(job) / self.bucket_files[bucket]
        path.write_text(json.dumps(data2, ensure_ascii=False, indent=2), encoding="utf-8")

    def sync_job_glossary(self, job) -> None:
        self.ensure_initialized(job)
        self._sync_job_glossary_noinit(job)

    def _sync_job_glossary_noinit(self, job) -> None:
        path = self.memory_dir(job) / self.bucket_files["glossary"]
        try:
            bucket = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            bucket = default_bucket("glossary")
        if not isinstance(bucket, dict):
            bucket = default_bucket("glossary")
        if not isinstance(bucket.get("entries"), list):
            bucket["entries"] = []
        job_glossary = getattr(job, "glossary", None)
        if isinstance(job_glossary, dict) and isinstance(job_glossary.get("entries"), list):
            if not bucket.get("entries"):
                bucket["entries"] = [x for x in job_glossary.get("entries", []) if isinstance(x, dict)]
                bucket["version"] = "0.1.0"
                bucket["bucket"] = "glossary"
                bucket["updated_at"] = utc_iso()
                path.write_text(json.dumps(bucket, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                job.glossary = {
                    "version": "0.1.0",
                    "entries": [x for x in bucket.get("entries", []) if isinstance(x, dict)],
                    "updated_at": bucket.get("updated_at") or utc_iso(),
                }
        else:
            job.glossary = {
                "version": "0.1.0",
                "entries": [x for x in bucket.get("entries", []) if isinstance(x, dict)],
                "updated_at": bucket.get("updated_at") or utc_iso(),
            }

    def _entry_identity(self, entry: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(entry.get("type") or ""),
            str(entry.get("src_text") or "").strip(),
            str(entry.get("tgt_text") or "").strip(),
            str(entry.get("scope") or "project"),
        )

    def upsert_entries(self, job, bucket: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []

        data = self.load_bucket(job, bucket)
        current = [x for x in data.get("entries", []) if isinstance(x, dict)]
        by_key = {self._entry_identity(x): idx for idx, x in enumerate(current)}
        added_or_updated: List[Dict[str, Any]] = []

        for raw in entries:
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            if not str(entry.get("src_text") or "").strip() and bucket != "style_memory":
                continue
            if not str(entry.get("tgt_text") or "").strip() and bucket in {"glossary", "translation_memory"}:
                continue
            entry.setdefault("entry_id", str(uuid.uuid4()))
            entry.setdefault("scope", "project")
            entry.setdefault("source", "system")
            entry["updated_at"] = utc_iso()
            key = self._entry_identity(entry)
            if key in by_key:
                old = current[by_key[key]]
                merged = {**old, **entry}
                current[by_key[key]] = merged
                added_or_updated.append(merged)
            else:
                current.append(entry)
                by_key[key] = len(current) - 1
                added_or_updated.append(entry)

        data["entries"] = current
        self.save_bucket(job, bucket, data)
        if bucket == "glossary":
            self.sync_job_glossary(job)
        return added_or_updated

    def add_style_hint(self, job, *, message: str, key: str, value: str) -> None:
        self.upsert_entries(
            job,
            "style_memory",
            [
                {
                    "type": "style",
                    "src_text": key,
                    "tgt_text": value,
                    "scope": "project",
                    "source": "user_message",
                    "note": message,
                }
            ],
        )

    def add_project_rule(self, job, *, message: str, key: str, value: str) -> None:
        self.upsert_entries(
            job,
            "project_memory",
            [
                {
                    "type": "rule",
                    "src_text": key,
                    "tgt_text": value,
                    "scope": "project",
                    "source": "user_message",
                    "note": message,
                }
            ],
        )

    def append_brain_event(
        self,
        job,
        *,
        event_type: str,
        message: str,
        summary: str,
        extra: Optional[Dict[str, Any]] = None,
        keep_last: int = 80,
    ) -> Dict[str, Any]:
        self.ensure_initialized(job)
        data = self.load_bucket(job, "brain_memory")
        entries = [x for x in data.get("entries", []) if isinstance(x, dict)]
        entry: Dict[str, Any] = {
            "entry_id": str(uuid.uuid4()),
            "type": str(event_type or "message"),
            "src_text": str(message or "").strip(),
            "tgt_text": str(summary or "").strip(),
            "scope": "brain",
            "source": "user_message",
            "updated_at": utc_iso(),
        }
        if isinstance(extra, dict):
            entry.update(extra)
        entries.append(entry)
        if keep_last > 0:
            entries = entries[-keep_last:]
        data["entries"] = entries
        self.save_bucket(job, "brain_memory", data)
        return entry

    def snapshot(self, job) -> Dict[str, Any]:
        self.ensure_initialized(job)
        snap: Dict[str, Any] = {}
        for bucket in self.bucket_files:
            snap[bucket] = self.load_bucket(job, bucket)
        return snap

    def export_rag_bundle(self, job) -> Dict[str, Any]:
        snap = self.snapshot(job)
        return {
            "version": "0.1.0",
            "bundle_type": "agent1_rag_bundle",
            "created_at": utc_iso(),
            "metadata": {
                "source_job_id": getattr(job, "job_id", None),
                "source_file_name": getattr(job, "source_file_name", None),
                "target_language": getattr(job, "target_language", None),
            },
            "buckets": {bucket: snap.get(bucket, default_bucket(bucket)) for bucket in self.rag_bundle_buckets},
        }

    def import_rag_bundle(self, job, bundle: Dict[str, Any]) -> Dict[str, int]:
        if not isinstance(bundle, dict):
            return {bucket: 0 for bucket in self.rag_bundle_buckets}

        buckets = bundle.get("buckets")
        if not isinstance(buckets, dict):
            buckets = {bucket: bundle.get(bucket) for bucket in self.rag_bundle_buckets if isinstance(bundle.get(bucket), dict)}

        imported: Dict[str, int] = {}
        for bucket in self.rag_bundle_buckets:
            raw = buckets.get(bucket) if isinstance(buckets, dict) else None
            entries = raw.get("entries") if isinstance(raw, dict) else None
            added = self.upsert_entries(job, bucket, [x for x in (entries or []) if isinstance(x, dict)])
            imported[bucket] = len(added)

        self.sync_job_glossary(job)
        return imported
