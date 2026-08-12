# Memory Retrieval Scaling Guide

This note defines a practical upgrade path for memory retrieval in the manga agent.
The goal is not theoretical perfection. The goal is to know when the current approach
is still good enough, and when complexity is actually worth paying for.

## 1. Current Recommendation

Use local structured retrieval first:

- storage: local JSON files
- retrieval: Python filtering + scoring
- no extra model call for retrieval
- no embedding database

For the current project stage, this should remain the default.

## 2. Retrieval Levels

### Level 0: Pure JSON + In-Memory Filtering

Use when:

- total memory entries are still small
- one retrieval only scans tens to low hundreds of entries
- retrieval latency is negligible compared with VLM / inpaint latency
- most matching can be done by exact fields or simple fuzzy scoring

Typical storage:

- `brain_memory.json`
- `translation_memory.json`
- `project_memory.json`
- `style_memory.json`
- `experience_memory.json`

Typical query style:

- filter by `stage`, `page_type`, `kind`, `recommended_action`
- filter by `series`, `issue`, `language_pair` when available
- rank a small candidate set
- return top 3-5 items only

Stay on Level 0 if these are true:

- total long-term memory entries < 2,000
- one bucket usually < 500 entries
- one retrieval pass stays under about 10-20 ms locally
- Brain prompt only needs a few retrieved items

For `agent1`, this is the recommended default right now.

### Level 1: Indexed Local SQLite

Upgrade to SQLite when JSON is still logically correct, but operationally starts to feel clumsy.

Use when:

- memory entries are growing steadily
- filtering logic gets more complex
- multiple files must be opened every time
- you want field indexes without bringing in vector retrieval

Move to Level 1 if two or more of these happen:

- total long-term memory entries > 2,000-5,000
- one bucket often > 1,000 entries
- retrieval frequently exceeds about 30-50 ms
- namespace filtering becomes common:
  - same series
  - same franchise
  - same user
  - same language pair
- you want dedupe / merge / analytics to be cleaner

What SQLite should store:

- one row per memory entry
- indexed columns:
  - `bucket`
  - `kind`
  - `stage`
  - `page_type`
  - `series`
  - `franchise`
  - `user_id`
  - `language_pair`
  - `recommended_action`
  - `updated_at`

What SQLite should not do yet:

- embeddings
- semantic reranking model calls
- extra retrieval LLM calls

SQLite is the best "next step" if memory becomes large but still mostly structured.

### Level 2: Hybrid Structured Retrieval + Embedding Search

Only upgrade here when exact fields are no longer enough.

Use when:

- user wording varies a lot
- terminology matching is no longer mostly exact
- you need semantic similarity across many historical corrections
- memory contains many natural-language notes that matter

Move to Level 2 only if these become real problems:

- relevant memory is often missed by structured filtering
- users frequently describe the same issue in different language
- exact matching produces low recall
- the number of free-text memories becomes large enough that manual scoring is weak

Recommended pattern:

- first do structured filtering
- then run embedding search only inside the filtered subset
- still return a very small top-k

Do not do:

- global vector search across everything on every step
- embedding retrieval for every page stage by default
- feeding large retrieval dumps into Brain

For `agent1`, this level is not recommended yet.

### Level 3: Full RAG Stack

This means a real retrieval stack:

- vector DB
- semantic rerank
- more advanced retrieval policies
- possibly separate retrieval services

Only use when:

- the project becomes multi-user or platform-level
- memory grows very large
- semantic search becomes a core product feature
- retrieval quality clearly limits output quality

This is overkill for the current local manga workflow.

## 3. Decision Rules For This Project

### Keep Pure JSON If

- the main latency is still VLM / erase / typeset
- retrieval work is only local file read + Python filtering
- retrieved memory per step is small
- debugging the memory files directly is useful

This is where the project is today.

### Upgrade To SQLite If

- memory becomes annoying to inspect or merge by hand
- JSON bucket files become large and noisy
- retrieval logic becomes more index-like than file-like
- you want safer updates and easier dedupe

This is the most likely future upgrade path.

### Consider Embeddings Only If

- translation corrections become semantically diverse
- users rely heavily on fuzzy natural-language reuse
- exact fields plus simple fuzzy scoring are no longer enough

This should be deferred as long as possible.

## 4. Practical Latency Budget

For this project, a good rule is:

- local retrieval target:
  - ideal: < 10 ms
  - acceptable: < 30 ms
  - warning: > 50 ms
- remote or model stage:
  - often hundreds of ms to seconds

So retrieval should remain a tiny fraction of total runtime.

If retrieval becomes noticeable to the user, it is already too heavy for the current architecture.

## 5. Namespace Guidance

Do not implement namespace by duplicating storage.

Preferred approach:

- keep one copy of each memory entry
- add scope metadata:
  - `series`
  - `issue`
  - `franchise`
  - `user_id`
  - `language_pair`
  - `pipeline_mode`
- use these fields to filter and score

This avoids storage blow-up and keeps retrieval cheap.

## 6. Retrieval Safety Rules

No matter which level is used:

- never retrieve everything
- always filter by stage first when possible
- prefer exact metadata over semantic search
- cap returned items to a small top-k
- keep retrieval model-free unless there is strong evidence you need semantics
- do not let memory retrieval become slower than the page step it is supporting

## 7. Recommendation For Agent1

Right now:

- stay on Level 0
- keep JSON buckets
- add metadata fields before adding new infrastructure
- continue using small top-k structured retrieval

Next likely upgrade:

- Level 1 SQLite

Not recommended yet:

- full embedding RAG
- vector DB
- external retrieval service

## 8. Simple Rule Of Thumb

Use this rule:

- If the problem is "which exact known memory applies?" -> structured filtering
- If the problem is "the memory is too big to scan cleanly" -> SQLite
- If the problem is "the right memory exists but exact filtering cannot find it" -> embeddings

That is the practical boundary.
