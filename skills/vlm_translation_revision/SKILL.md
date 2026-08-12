# VLM Translation Revision Skill

## Purpose

Revise translated text without rerunning the VLM image step.

This skill works on a text document envelope:

- top-level `document_range`
- `pages[]`
- `regions[]`

It is used both for:

- `auto_rag_revision`
- `user_feedback_revision`

---

## Input Contract

Required fields:

- `document_range`
  - `start_page_index`
  - `end_page_index`
  - `page_count`
  - `page_indices`
- `pages`
  - each page item contains:
    - `page_index`
    - `page_type`
    - `regions`
      - `region_id`
      - `bbox_xyxy`
      - `has_text_box`
      - `color`
      - `src_text`
      - `tgt_text`

Other important fields:

- `revision_mode`
  - `auto_rag_revision`
  - `user_feedback_revision`
- `user_request`
- `reference_page_number`
- `scope`
- `glossary`
- `retrieval_context`
  - may now contain window-level context:
    - `document_range`
    - `target_page_indices`
    - `target_page_examples`
    - `window_context`

---

## Output Contract

```json
{
  "decision": "apply",
  "document_range": {
    "start_page_index": 0,
    "end_page_index": 9,
    "page_count": 10,
    "page_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "label": "0-9"
  },
  "pages": [],
  "changed_pages": [1, 2],
  "patches": [],
  "glossary_updates": [],
  "warnings": []
}
```

Notes:

- The skill should preserve passthrough metadata like `page_type`, `has_text_box`, and `color`.
- The skill must not mutate `region_id` or `bbox_xyxy`.
- The skill only patches `tgt_text`.
- The returned `document_range` should match the input window so downstream stages know which page bundle was revised.

---

## Brain / Orchestrator Guidance

- Use `auto_rag_revision` for lightweight consistency fixing after VLM output.
- Use `user_feedback_revision` when the user provides a correction instruction.
- Keep candidate pages bounded; do not send the entire book when only one window is relevant.
- If a page already has a valid erased cache, revision should reuse that cache later instead of forcing new inpainting.
- For `auto_rag_revision`, prefer a full VLM window when the translated pages came from one VLM batch and all still need consistency alignment.
