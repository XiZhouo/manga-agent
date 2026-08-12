# Script inventory

The project keeps scripts in four small groups. Generated artifacts belong under
`data/test_runs/` and should not be committed.

## Primary entry points

- `chat_brain_cli.py`: interactive Brain/chat entry point.
- `test_live_pipeline_pages.py`: canonical live VLM -> erase -> typeset regression.
- `test_nightwing_vlm_live.py`: VLM-only live regression on selected pages.

## Remote service checks and operators

- `test_remote_inpaint_service_live.py`: remote inpaint smoke test.
- `test_lama_server_smoke.py`: local LaMa server smoke test.
- `test_lettering_with_local_lama_server.py`: local lettering + LaMa integration.
- `test_manga_cleaner_page002_live.py`: detailed remote detector/inpaint tuning harness.
- `run_remote_inpaint_from_mask.py`: inpaint a saved image/mask pair.
- `run_remote_sam_refine_and_inpaint.py`: SAM-refine then inpaint a saved case.

## Offline regressions

Files ending in `_offline.py`, plus `test_vlm_quality_retry.py`,
`test_vlm_hybrid_judge_retry.py`, and `test_max_steps_autoscale_classic.py`, cover
orchestrator decisions, QA, batching, revision, retrieval, memory, and manifests.
They must not require model APIs or remote Windows services.

Run them from the repository root with the project interpreter and root on
`PYTHONPATH`:

```bash
PYTHONPATH=. .venv/bin/python scripts/test_job_level_vlm_batching_offline.py
```

## Maintenance rule

Do not add scripts tied to a single historical output directory. Extend a
parameterized regression above, or keep short-lived diagnostics outside the
repository. Preserve only representative successful and failure-attribution
artifacts under `data/test_runs/`.
