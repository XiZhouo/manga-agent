# SQLite Shared Memory Design

This document defines how `agent1` should evolve from job-local JSON memory
into a mixed architecture:

- job-local JSON for task state and artifacts
- shared SQLite memory for cross-job reusable knowledge

The goal is practical engineering value:

- preserve job isolation
- reuse user-confirmed translation knowledge across jobs
- support better auditability
- keep retrieval fast and structured
- remain compatible with the current codebase during migration

## 1. Why We Need This

Current state:

- each job has its own `memory/` directory
- glossary / translation memory / experience memory are job-local
- cross-job reuse only works by exporting and importing a RAG bundle manually

This is good for early iteration, but limited for product-like behavior.

Desired behavior:

- page masks / images / manifests stay isolated per job
- reusable translation knowledge can survive across jobs
- user corrections in one comic can help future jobs when the scope is appropriate
- technical experience can guide future parameter choices

## 2. Architecture Overview

Keep two layers.

### 2.1 Job Layer

Still stored on filesystem:

- job state
- page state
- skill outputs
- masks
- erased images
- rendered images
- manifests
- logs
- optional job-local JSON memory as transition layer

Why:

- large artifacts belong on disk
- job replay and debugging stay simple
- page-level state must remain isolated

### 2.2 Shared Memory Layer

Stored in SQLite:

- glossary
- translation memory
- style memory
- project rules
- experience memory
- optional later: persistent chat/session memory

Recommended DB path:

- `data/memory/shared_memory.db`

This DB is the reusable knowledge layer across jobs.

## 3. Core Design Principle

Do not copy memory between jobs.

Instead:

- store one canonical shared record
- attach scope metadata
- retrieve by scope priority

This avoids both storage blow-up and retrieval noise.

## 4. Scope Model

Each shared memory entry should have a scope.

Recommended fields:

- `scope_type`
- `scope_value`

Suggested scope types:

- `global`
- `franchise`
- `series`
- `issue`
- `session`
- `job`

Examples:

- `global` + `null`
  - truly universal rule
- `franchise` + `dc`
  - broad domain memory
- `series` + `batman`
  - reusable only for Batman-related jobs
- `issue` + `batman_001`
  - one specific issue
- `session` + `<session_id>`
  - current chat session only
- `job` + `<job_id>`
  - current task only

## 5. Which Memory Should Be Shared

### Share By Default

- `glossary`
- `translation_memory`
- `style_memory`
- `project_memory`
- `experience_memory`

### Keep Local By Default

- `brain_memory`
- page intermediate decisions
- raw skill artifacts
- job logs

Reason:

- shared layer should store reusable knowledge
- local layer should store temporary execution history

## 6. Table Design

## 6.1 `memory_entries`

Main table for shared knowledge.

Suggested columns:

- `id` INTEGER PRIMARY KEY
- `entry_uuid` TEXT UNIQUE NOT NULL
- `memory_type` TEXT NOT NULL
  - `glossary | translation | style | project | experience`
- `scope_type` TEXT NOT NULL
- `scope_value` TEXT
- `source_text` TEXT
- `target_text` TEXT
- `note` TEXT
- `source_kind` TEXT NOT NULL
  - `user_revision | auto_revision | runtime_observation | import_bundle | user_message | system`
- `glossary_type` TEXT
  - `term | quote | name | phrase`
- `recommended_action` TEXT
- `stage` TEXT
- `page_type` TEXT
- `language_pair` TEXT
  - example: `en->zh`
- `series_key` TEXT
- `issue_key` TEXT
- `source_file_name` TEXT
- `locked` INTEGER NOT NULL DEFAULT 0
- `confidence` REAL
- `skill_params_json` TEXT
- `tags_json` TEXT
- `extra_json` TEXT
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL
- `last_used_at` TEXT
- `use_count` INTEGER NOT NULL DEFAULT 0

Recommended indexes:

- `(memory_type, scope_type, scope_value)`
- `(memory_type, series_key)`
- `(memory_type, language_pair)`
- `(recommended_action)`
- `(stage, page_type)`
- `(updated_at)`

## 6.2 `memory_aliases`

Optional alias table for terminology matching.

Columns:

- `id` INTEGER PRIMARY KEY
- `entry_uuid` TEXT NOT NULL
- `alias_text` TEXT NOT NULL

Use for:

- character aliases
- alternate spellings
- abbreviations

## 6.3 `memory_usage_log`

Audit trail for retrieval usage.

Columns:

- `id` INTEGER PRIMARY KEY
- `entry_uuid` TEXT NOT NULL
- `job_id` TEXT
- `page_index` INTEGER
- `route` TEXT NOT NULL
  - `vlm | revision | technical_decision | chat | progress`
- `reason` TEXT
- `used_at` TEXT NOT NULL

This is important for debugging and demo value.

## 6.4 `sessions` (Optional Later)

Only needed for persistent casual chat memory.

Not required in phase 1.

## 7. Mapping From Current JSON Buckets

### `glossary.json`

Map to:

- `memory_type = glossary`

Important fields:

- `source_text`
- `target_text`
- `glossary_type`
- `locked`
- `aliases_src`

### `translation_memory.json`

Map to:

- `memory_type = translation`

Important fields:

- `source_text`
- `target_text`
- `why`
- `page_index`
- `region_id`

### `style_memory.json`

Map to:

- `memory_type = style`

### `project_memory.json`

Map to:

- `memory_type = project`

### `experience_memory.json`

Map to:

- `memory_type = experience`

Important fields:

- `recommended_action`
- `stage`
- `page_type`
- `skill_params_json`
- `repeat_count`
- `confidence`

## 8. Write Rules

Not every event should become shared memory.

## 8.1 Glossary Write Rules

Write to shared DB when:

- user explicitly corrects a name or term
- user confirms a preferred translation
- imported RAG bundle contains reusable terminology

Default scope:

- `series` for names / recurring terms
- `issue` for issue-specific references

## 8.2 Translation Memory Write Rules

Write to shared DB when:

- user explicitly corrects a line or phrase
- auto revision produces a high-confidence reusable correction and policy allows saving

Default scope:

- `issue` or `job`

Promote to `series` only when:

- repeated correction appears multiple times
- or user explicitly says it should be reused broadly

## 8.3 Experience Write Rules

Write when:

- a failure pattern repeats
- a strategy is clearly successful
- a retry policy becomes stable

Default scope:

- `series` if strongly title-specific
- otherwise `global` or `franchise`

## 9. Retrieval Rules

Retrieval should stay route-aware.

## 9.1 VLM Route

Prefer:

- `glossary`
- `style`
- `project`

Only lightly use:

- `translation`

Reason:

- first-pass VLM translation should not be overloaded by too many line-level patches

Scope priority:

- `issue`
- `series`
- `franchise`
- `global`

## 9.2 Revision Route

Prefer:

- `translation`
- `glossary`
- `style`
- `project`

Reason:

- revision is the best place to apply line-level correction memory

## 9.3 Technical Decision Route

Prefer:

- `experience`
- recent job-local brain memory

Reason:

- this route is about retry choice and parameter selection

## 9.4 Chat Route

Prefer:

- recent session memory
- recent job-local brain memory
- maybe a small amount of shared style/project preference

Do not dump technical memory into normal chat unless relevant.

## 10. Retrieval Strategy

Phase 1 should remain structured retrieval.

Recommended pipeline:

1. filter by route and `memory_type`
2. filter by scope candidates
3. filter by language pair when relevant
4. score by:
   - exact source/target match
   - alias hit
   - tag hit
   - stage/page_type match
   - confidence
   - recency / use_count
5. return small top-k only

Recommended top-k:

- VLM: `3-8`
- revision: `5-12`
- technical decision: `3-6`
- chat: `3-5`

## 11. FTS / Embedding Strategy

Do not start with embeddings.

### Phase 1

- SQLite
- structured filtering
- optional simple fuzzy score in Python

### Phase 2

- SQLite FTS5 for text fields
- still no embedding dependency

### Phase 3

- optional hybrid retrieval with embeddings
- only after structured retrieval proves insufficient

If embeddings are added later:

- use them only inside a pre-filtered candidate set
- do not run semantic search over everything by default

## 12. Coexistence With Current JSON Memory

Migration should be incremental.

### Phase A

Keep:

- current job-local JSON buckets

Add:

- shared SQLite DB

Write path:

- continue writing local JSON
- also write selected reusable entries into SQLite

Read path:

- retrieval first reads shared SQLite
- then merges job-local JSON memory

### Phase B

Once stable:

- keep JSON only for local task state and easy inspection
- treat SQLite as the source of truth for shared reusable memory

### Phase C

Optional later:

- stop storing duplicated shared entries in job-local JSON
- keep JSON snapshots only for export/debug

## 13. Cross-Job Reuse Policy

This answers the key product question:

If the user corrected one term in comic A, should comic B use it?

Answer:

- yes, only if the scope allows it

Examples:

- same series:
  - yes
- same franchise:
  - maybe, lower priority
- unrelated comic:
  - usually no, unless scope is `global`

This should be enforced by retrieval ranking, not by manual export/import only.

## 14. Session Behavior

In one chat session, the user may run multiple jobs.

Desired behavior:

- each job keeps isolated artifacts and local state
- shared SQLite memory allows corrections from earlier jobs to help later jobs
- session-local chat continuity still comes from in-memory ring buffer first

Persistent cross-session chat memory is optional and should be separate from translation memory.

## 15. Why This Is Good For Engineering Credibility

This design is a strong resume/interview story because it shows:

- separation of execution state and reusable knowledge
- controlled cross-task memory reuse
- auditability through usage logs
- scoped retrieval rather than naive global memory injection
- incremental migration from local JSON to database-backed memory

This is more credible than jumping straight to a buzzword-heavy vector stack.

## 16. Implementation Order

Recommended order:

1. create shared SQLite schema
2. implement write helpers for:
   - glossary
   - translation memory
   - experience memory
3. implement route-aware read helpers
4. merge SQLite results with current job-local memory retrieval
5. add usage logging
6. only later consider FTS5
7. only later consider embeddings

## 17. Phase 1 Minimum Scope

If we want the smallest useful milestone, do only this:

- create `shared_memory.db`
- add `memory_entries`
- support 3 shared memory types:
  - `glossary`
  - `translation`
  - `experience`
- write on:
  - user revision
  - successful strategy
  - repeated failure pattern
- read on:
  - `vlm`
  - `revision`
  - `technical_decision`

That already solves the biggest current limitation:

- cross-job inheritance of user-confirmed memory

## 18. What We Do Not Need Yet

Not needed in phase 1:

- LangGraph migration
- vector DB
- BGE embeddings
- reranker service
- multi-agent memory bus

Those can be discussed later if retrieval quality truly becomes the bottleneck.
