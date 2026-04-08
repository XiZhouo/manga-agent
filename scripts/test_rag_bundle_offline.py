from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.memory_store import LocalMemoryStore
from orchestrator.tools import Tools


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_bundle_") as td:
        job_dir = Path(td)
        store = InMemoryStateStore()
        agent = ComicAgent(store=store, tools=Tools(), decision_engine=DecisionEngine())
        memory = LocalMemoryStore()

        job_id = agent.create_job(
            user_id="test",
            source_file_name="bundle_src.cbz",
            target_language="zh",
            pages=[(0, str(job_dir / "page_0000.png"))],
        )
        job = store.get_job(job_id)
        job.user_preferences["job_dir"] = str(job_dir)
        memory.upsert_entries(
            job,
            "glossary",
            [
                {
                    "type": "term",
                    "src_text": "Batman",
                    "tgt_text": "蝙蝠侠",
                    "locked": True,
                    "scope": "project",
                    "source": "seed",
                }
            ],
        )
        memory.upsert_entries(
            job,
            "translation_memory",
            [
                {
                    "type": "phrase",
                    "src_text": "Gotham City",
                    "tgt_text": "哥谭市",
                    "locked": False,
                    "scope": "project",
                    "source": "seed",
                }
            ],
        )
        store.save_job(job)

        bundle = memory.export_rag_bundle(job)
        bundle_path = job_dir / "rag_bundle.json"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print("bundle_buckets=", sorted((bundle.get("buckets") or {}).keys()))
        assert "brain_memory" not in (bundle.get("buckets") or {})

        job_id2 = agent.create_job(
            user_id="test",
            source_file_name="bundle_dst.cbz",
            target_language="zh",
            pages=[(0, str(job_dir / "page_0001.png"))],
        )
        job2 = store.get_job(job_id2)
        job2.user_preferences["job_dir"] = str(job_dir / "job2")
        imported = memory.import_rag_bundle(job2, json.loads(bundle_path.read_text(encoding="utf-8")))
        store.save_job(job2)

        print("imported_counts=", imported)
        snap = memory.snapshot(job2)
        glossary_entries = (snap.get("glossary") or {}).get("entries") or []
        tm_entries = (snap.get("translation_memory") or {}).get("entries") or []
        assert any(e.get("src_text") == "Batman" and e.get("tgt_text") == "蝙蝠侠" for e in glossary_entries)
        assert any(e.get("src_text") == "Gotham City" and e.get("tgt_text") == "哥谭市" for e in tm_entries)
        print("glossary_entries=", len(glossary_entries))
        print("translation_memory_entries=", len(tm_entries))


if __name__ == "__main__":
    main()
