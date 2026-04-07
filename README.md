# Manga Translation Agent (WIP)

This repo is building a workflow-driven Comic/Manga translation agent.

The orchestrator is the source of truth for execution + retries; “skills” are pure capability modules.

## Quickstart (dev)

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Runtime notes

- `pdf2image` requires Poppler (`pdftoppm`).
- `rarfile` may require `unrar` for `.rar/.cbr`.

## Design Spec (working notes)

This section is written as an implementation spec (not user documentation).

### Product constraints

- Primary translation: **English → Chinese**.
- Japanese: keep interfaces only (no implementation in v0).
- Reading order is a **user option**; default is **English LTR**.
- SFX/stylized lettering is **out-of-scope** for v0; keep extension points.
- Corrections happen **within the same job** (no cross-session correction UX).
- Packaging output uses **ZIP** (RAR mentioned earlier only meant “archive”).

### Canonical identifiers (must be carried across all steps)

All skills MUST preserve these identifiers so we can re-run minimal subsets:

```text
job_id: string
page_index: int   (0-based)
region_id: int    (bubble id, unique within a page)
```

Optional stable ids for future speaker alignment:

```text
utterance_id: string  (stable within job; derived from page_index+region_id)
speaker_id: string | null
```

### Workflow (orchestrator-driven)

```text
Classic (fallback)
	InputNormalization
		-> PanelTextDetection
			-> RegionOCR
				-> ScriptBuilder(v0)
					-> Translation(en->zh, glossary-aware)
						-> Lettering(v0, block text)
							-> PackageZIP

VLM core (new)
	InputNormalization
		-> VLMPageTranslate(qwen3.6-plus: detect+OCR+translate)
			-> Lettering(v0, block text)
				-> PackageZIP
```

Switching policy:

- Set `job.user_preferences.pipeline_mode` to `classic` or `vlm` (CLI supports `--pipeline-mode`).

Token/quality knob (VLM):

- `MANGA_AGENT_VLM_MAX_HEIGHT` (default: `3057`). If the image height is larger, it will be downscaled for the VLM call, and returned bboxes are mapped back to original pixel coordinates.
- `MANGA_AGENT_VLM_MAX_WIDTH` (default: `0`, meaning no cap). If set (e.g. `3057`), width larger than this will be downscaled too (cheaper but may affect double-page spreads).
- `MANGA_AGENT_VLM_MAX_LONG_SIDE` (default: `0`, optional extra safety cap).

Fallback policy:

- If the first VLM call was downscaled and returned **0 regions**, the orchestrator retries once with full resolution (costs more tokens but avoids missing-page failures).

Hybrid QA policy (VLM):

- **Obvious failures** are handled by the heuristic gate (empty texts, wrong language, 0 regions but detector found text), and may trigger an automatic VLM retry (budgeted).
- **Non-obvious cases** can be evaluated by a **text-only LLM judge** (no image upload). The judge receives region `src_text/tgt_text` samples + summary metrics, and returns `proceed|retry|ask_user`.

Judge knobs (text-only):

- `MANGA_AGENT_VLM_JUDGE_MODE` = `off|auto|on` (default: `auto`)
- `MANGA_AGENT_VLM_JUDGE_MODEL` (default: `qwen3-max`)
- `MANGA_AGENT_VLM_JUDGE_TEMPERATURE` (default: `0.0`)
- `MANGA_AGENT_VLM_JUDGE_MAX_REGIONS` (default: `12`)
- `MANGA_AGENT_VLM_JUDGE_MAX_TEXT_LEN` (default: `280`)

### Route A: “Agent brain” + text-only revision

To behave like an **agent** (not just a fixed pipeline), the orchestrator supports a Route-A loop:

1) User sends feedback/correction in natural language.
2) A **DecisionBrain (LLM)** parses it into a structured plan (`job.pending_user_plan`).
3) The DecisionEngine executes the plan (ask user / rerun page / run revision).
4) For corrections, we run a **text-only revision skill** that patches existing VLM outputs (no image upload, no VLM rerun).
5) Only affected pages are re-lettered.

Brain knobs:

- `MANGA_AGENT_BRAIN_MODE` = `off|auto` (default: `auto`)
- `MANGA_AGENT_BRAIN_MODEL` (default: `qwen3-max`)
- `MANGA_AGENT_BRAIN_TEMPERATURE` (default: `0.0`)

Revision knobs (text-only):

- `MANGA_AGENT_REVISION_MODEL` (default: `qwen3-max`)
- `MANGA_AGENT_REVISION_TEMPERATURE` (default: `0.0`)
- `MANGA_AGENT_REVISION_MAX_CANDIDATE_PAGES` (default: `12`)
- `MANGA_AGENT_REVISION_MAX_PATCHES` (default: `64`)

LLM responsibilities (v0):

- Translate with glossary constraints.
- Judge (text-only) for non-obvious quality issues.
- Parse user corrections into a plan (DecisionBrain).
- Apply text-only translation revision (patch `tgt_text` + emit `glossary_updates`).

### Job lifecycle (single-job correction loop)

The job runs until it reaches a “review/correction loop”:

```text
RUN (produce draft outputs)
	-> WAITING_USER (review/correct)
		-> (apply corrections) -> rerun MINIMAL affected regions
		-> WAITING_USER
	-> (user confirms export) -> COMPLETED + ZIP output
```

User-facing correction inputs we support:

1) Upload a glossary file update.
2) Natural language message like:
	 - “希望力量陪着你 翻译错了，应该是 原力与你同在”

Correction MUST NOT require users to provide `page_index/region_id`.

### Artifacts (internal + returned)

Even if correction is “within job”, we still generate editable artifacts so users can adjust terms:

- `glossary.json` (editable by user; re-upload to update within job)
- `run_manifest.json` (internal index; used to map text -> page/bubble)

NOTE: `run_manifest.json` should be considered internal; it is allowed to be packaged but is not meant as user UX.

### Glossary / RAG (v0 contract)

We implement “RAG” as a pluggable retrieval layer. v0 can be exact/alias matching;
later we can swap retrieval backend to vector DB without changing the translation I/O.

`glossary.json` example:

```json
{
	"version": "0.1.0",
	"entries": [
		{
			"entry_id": "e_force_001",
			"type": "quote",
			"src_text": "may the force be with you",
			"tgt_text": "原力与你同在",
			"aliases_src": ["May the Force be with you"],
			"aliases_tgt": ["希望力量陪着你"],
			"locked": true,
			"updated_at": "2026-04-04T00:00:00Z"
		}
	]
}
```

Key behaviors:

- Auto-learned entries are `locked=false` by default.
- User-edited entries are `locked=true`.
- When user corrects by Chinese text, we must be able to find the relevant entry/occurrence via:
	- `aliases_tgt` and
	- `run_manifest.json` reverse index (translated_text -> page_index/region_id)

### Skill contracts (v0)

All skills should use a consistent envelope:

```text
{
	ok: bool,
	data: <payload>,
	metrics: { ... },
	trace: { job_id, page_index?, region_id? },
	error: { code, message, detail? } | null
}
```

#### Skill0: input_normalization

Input:

```text
file_bytes, file_type
```

Output payload:

```text
pages: [{page_index, image_bytes, filename}]
```

#### Skill1: panel_text_detection

Input:

```text
page_index, image_bytes, model_choice, conf_threshold, iou_threshold, imgsz
```

Output payload:

```text
regions: [{region_id, bbox_xywh, score, region_type}]
image_size: {width, height}

#### Skill1b: vlm_page_translate (VLM core)

Input:

```text
page_index, image_bytes, reading_direction, source_language=en, target_language=zh
```

Output payload:

```text
regions: [{region_id, bbox_xywh, src_text, tgt_text}]
image_size: {width, height}
```
```

#### Skill2: region_ocr

Input:

```text
page_index, image_bytes, regions[bbox_xywh], source_language=en
```

Output payload:

```text
results: {region_id: {text, confidence?, flags[]}}
```

#### Skill3: script_builder_v0

Input:

```text
page_index, regions[{region_id, bbox_xywh, ocr_text}], reading_direction
```

Output payload:

```text
utterances: [{order_index, utterance_id, region_id, clean_text, speaker_id?}]
page_context: string
```

#### Skill4: translate_script_en2zh

Input:

```text
utterances, page_context, glossary_context, character_profiles?
```

Output payload:

```text
results: {region_id: {translated_text, confidence?}}
memory_updates: [{entry_id?, src_text, tgt_text, aliases_tgt?}]
conflicts: [{src_text, candidates[]}] 
```

#### Skill5: lettering_basic_v0

Input:

```text
image_bytes, regions[{region_id, bbox_xywh, translated_text}], render_params_hint?
```

Output payload:

```text
rendered_image_bytes
results: {region_id: {render_ok, overflow, render_params}}
```

#### Skill6: package_zip

Input:

```text
pages[{page_index, image_bytes, filename}], glossary.json, run_manifest.json
```

Output payload:

```text
zip_bytes, zip_filename
```

### Implementation milestones (next)

1) Fix skill defaults (detection model paths + imgsz=1280).
2) Add LangChain LLM adapter using DashScope OpenAI-compatible endpoint (Qwen).
3) Implement `translate_script_en2zh` (v0) with glossary context + JSON output.
4) Move `main.py` to orchestrator-driven CLI (thin wiring only).
5) Add VLM core workflow (qwen3.6-plus) + `pipeline_mode` switch.

