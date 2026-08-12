# Memory / RAG Architecture Checklist

This document defines the recommended memory layout for the manga agent.
It is intentionally practical: the goal is to help future implementation and review,
not to be a generic LLM memory essay.

## 1. Design Goals

- Keep Brain conversational like a normal assistant.
- Preserve task continuity across long manga jobs.
- Reuse confirmed translation corrections across pages, issues, and sessions.
- Avoid polluting task reasoning with irrelevant casual chat.
- Keep memory write rules explicit so the agent does not over-save noisy data.
- Keep retrieval targeted so Brain sees the right information at the right step.

## 2. Core Principle

Treat memory as two separate concerns:

- Storage: what is saved, when it is saved, how long it lives, and where it is stored.
- Retrieval: what is selected and injected back into Brain for the current step.

In practice:

- Short-term memory is mostly read by direct state lookup.
- Long-term memory is mostly read by exact filters plus lightweight RAG retrieval.
- Not every memory bucket needs vector search.

## 3. Memory Layers

The project should use four layers.

### 3.1 Chat Session Memory

Purpose:

- Preserve conversational continuity for Brain in the current CLI session.
- Handle follow-up phrases like "again", "continue", "what about that one", and "you just said...".

Scope:

- Session-local by default.
- Later can be upgraded to cross-session chat memory.

Store:

- In-memory ring buffer first.
- Optional later persistence: `data/chat_sessions/<session_id>.json` or SQLite.

Recommended fields:

```json
{
  "turn_id": "uuid",
  "timestamp": "iso8601",
  "user_text": "string",
  "assistant_text": "string",
  "intent": "chat|create_translation_job|query_progress|...",
  "active_job_id": "string|null"
}
```

Write when:

- Every normal Brain chat turn.
- Also after job-management replies if the exchange matters conversationally.

Read when:

- Always for normal Brain chat replies.
- Usually last 4-8 turns are enough.

Do not store:

- Full model raw payloads.
- Very large copied artifacts.

### 3.2 Job Memory

Purpose:

- Preserve execution context for one manga translation task.
- Let Brain explain what happened, why it failed, and what should happen next.

Scope:

- Bound to one `job_id`.

Store:

- Existing job directory and state system.
- Main references:
  - `run_manifest.json`
  - `memory/brain_memory.json`
  - page state
  - skill outputs

Recommended content:

- User requests that changed task behavior.
- Brain decisions and rationale.
- Per-page failures and retry outcomes.
- Important stage summaries, not every tiny log line.

Recommended fields for `brain_memory` entries:

```json
{
  "entry_id": "uuid",
  "updated_at": "iso8601",
  "type": "chat|decision|failure|revision|operator_note",
  "scope": "brain|job|page",
  "job_id": "string",
  "page_index": 2,
  "src_text": "user message or trigger",
  "tgt_text": "short summary for retrieval",
  "plan": {},
  "note": "short operator-facing note",
  "tags": ["erase", "timeout", "remote_service"]
}
```

Write when:

- Brain makes a non-trivial decision.
- A page fails or becomes blocked.
- A revision is applied.
- User explicitly asks for a rerun, correction, or workflow change.

Read when:

- Querying progress.
- Explaining failure.
- Deciding rerun/retry strategy.
- Resuming an interrupted job.

### 3.3 Project / Translation Memory

Purpose:

- Persist reusable translation knowledge beyond a single page.
- Reuse confirmed terminology and style decisions in later tasks.

Scope:

- Usually one project, one series, or one user + series namespace.

Current buckets already fit this idea:

- `glossary`
- `translation_memory`
- `project_memory`
- `style_memory`

Recommended responsibilities:

- `glossary`
  - High-confidence terminology pairs.
  - Canon names, titles, repeated catchphrases.
- `translation_memory`
  - Confirmed patch history from revision steps.
  - "X should be translated as Y" records.
- `project_memory`
  - Rules for this title or pipeline.
  - Example: "Nightwing uses names transliterated, not free localized."
- `style_memory`
  - Rendering and style preferences.
  - Example: "Keep SFX punchy", "favor concise dialogue bubbles".

Recommended fields for terminology-style memory:

```json
{
  "entry_id": "uuid",
  "updated_at": "iso8601",
  "source": "user_revision|auto_revision|operator_confirmed",
  "confidence": 0.95,
  "applies_to": {
    "user_id": "string|null",
    "series": "string|null",
    "source_file_name": "string|null",
    "language_pair": "en->zh"
  },
  "src_text": "string",
  "tgt_text": "string",
  "note": "why this matters",
  "tags": ["character_name", "catchphrase", "style_rule"]
}
```

Write when:

- User explicitly corrects a translation.
- User confirms a style or terminology preference.
- Agent completes an auto-revision with sufficiently high confidence and explicit policy allows saving.

Read when:

- VLM translation pre-context.
- Revision self-check.
- Brain deciding whether to revise or trust current output.

### 3.4 Experience Memory

Purpose:

- Help Brain reason about pipeline strategy, not just language output.
- Capture recurring operational lessons.

Examples:

- "Remote erase service timeout is infra, not mask-quality."
- "For embedded art text, art_embed lettering is usually better."
- "For Nightwing page002, hybrid mask v9 worked."

Scope:

- Cross-job technical memory.
- Mostly internal to the agent, not user-facing.

Store:

- Can start inside `project_memory` or a new bucket later such as `experience_memory`.

Recommended fields:

```json
{
  "entry_id": "uuid",
  "updated_at": "iso8601",
  "kind": "failure_pattern|successful_strategy|operator_rule",
  "situation": {
    "page_profile": "boxed_dialogue|art_integrated",
    "stage": "erase|typeset|vlm",
    "error_code": "detect_text_mask_timeout"
  },
  "recommended_action": "ask_user|retry_erase|retry_typeset",
  "skill_params": {},
  "note": "short explanation"
}
```

Write when:

- The same failure pattern appears multiple times.
- A strategy has been manually verified as successful.

Read when:

- Brain parameter suggestion.
- Retry policy.
- Failure explanation.

## 4. Retrieval Strategy

Use different retrieval methods for different buckets.

### 4.1 Direct Lookup

Best for:

- Current job state
- Current page state
- Recent chat turns
- Exact glossary records

Use direct lookup when:

- The key is already known.
- Freshness matters more than fuzzy similarity.

### 4.2 Metadata Filtering

Best for:

- Translation memory
- Project memory
- Revision history

Recommended filters:

- `user_id`
- `series`
- `source_file_name`
- `language_pair`
- `job_id`
- `page_index`
- `tags`
- `stage`

### 4.3 Lightweight RAG / Semantic Retrieval

Best for:

- Large correction history
- Cross-job style examples
- Similar failure patterns
- Experience memory

Recommended use:

- Only after metadata filtering narrows the candidate pool.
- Return a small top-k set, usually 3-8 snippets.

Do not use RAG for:

- Current page state
- Exact artifact paths
- Exact job/page identifiers

## 5. What Brain Should Read By Scenario

### 5.1 Normal Chat

Read:

- Recent chat session memory
- Optional user profile memory

Do not read:

- Full task memory unless user is asking about a job

### 5.2 Create Translation Job

Read:

- User profile and project-level preferences
- Optional imported RAG bundle metadata

### 5.3 VLM Translation

Read:

- Glossary
- Translation memory
- Project memory
- Possibly a tiny amount of style memory

Do not read:

- Large casual chat history
- Unrelated failure history

### 5.4 Revision

Read:

- Glossary
- Translation memory
- Current page or batch regions
- Recent revision patches

This is the most RAG-sensitive step in the current system.

### 5.5 Erase / Typeset Retry

Read:

- Current page QA
- Current page artifacts and errors
- Experience memory for similar technical failures

This should be mostly structured retrieval, not semantic text retrieval.

### 5.6 Progress / Resume

Read:

- Job memory
- Recent brain decisions
- Recent page failures
- Artifact refs from manifest

## 6. Write Policy

Not everything should become memory.

### Save immediately

- Explicit user correction
- Explicit user style preference
- Blocking pipeline error
- Brain decision that changes workflow

### Save only after confirmation or high confidence

- Auto-revision terminology updates
- Derived style assumptions
- Successful parameter strategy that should generalize

### Do not save

- Generic chit-chat unless needed for session continuity
- One-off noise
- Large raw tool payloads
- Duplicate entries with no new information

## 7. Separation Between Chat Memory and Task Memory

Keep them logically separate.

Why:

- Chat memory is high-noise and short-lived.
- Task memory must stay precise and auditable.
- Mixing them hurts retrieval quality.

Recommended namespaces:

- `chat/session/<session_id>`
- `chat/profile/<user_id>`
- `job/<job_id>`
- `project/<project_key>`
- `series/<series_key>`

They may share the same storage engine later, but the schema and retrieval policy should remain separate.

## 8. Storage Recommendation For This Project

Do not over-engineer yet.

### Phase 1: Current-friendly setup

- JSON files per bucket
- Existing job directory structure
- Existing manifest and skill output directories
- In-memory recent chat buffer

### Phase 2: Better cross-session memory

- SQLite for chat sessions and user profile
- JSON remains acceptable for job artifacts

### Phase 3: Optional retrieval upgrade

- Add vector index only for:
  - translation memory
  - project memory
  - experience memory

Do not vectorize everything.

## 9. Immediate Implementation Checklist

This is the recommended build order.

### A. Short-term chat continuity

- [x] Recent in-session chat turn buffer in `ChatController`
- [ ] Optional cross-session chat history store
- [ ] Optional chat-summary compaction after long sessions

### B. Task memory robustness

- [x] `brain_memory` bucket exists
- [x] job/page artifacts stored under job directory
- [x] manifest has navigable artifact refs
- [ ] Make Brain read more of recent job memory during progress/resume
- [ ] Add structured "operator note" entries for manual interventions

### C. Translation memory / glossary

- [x] glossary bucket exists
- [x] revision updates can be persisted
- [ ] Add namespace strategy for series/project-level reuse
- [ ] Add confidence and source fields consistently across saved entries

### D. Retrieval router

- [ ] Build one retrieval router function that chooses which buckets to query for each Brain intent
- [ ] Return both structured exact hits and optional semantic hits
- [ ] Keep top-k very small

### E. Experience memory

- [ ] Start saving recurring technical lessons
- [ ] Save successful verified strategies only
- [ ] Feed them into retry suggestion logic

## 10. Recommended Final Mental Model

Use this model when making future design decisions:

- Chat session memory keeps conversation smooth.
- Job memory keeps execution coherent.
- Project/translation memory keeps translations consistent.
- Experience memory keeps the pipeline smarter over time.
- RAG is not the memory itself; RAG is one retrieval tool used mainly for long-term memory.

If there is any conflict:

- Prefer precise structured retrieval over broad semantic retrieval.
- Prefer task memory cleanliness over saving everything.
- Prefer small, targeted context injection over large memory dumps.
