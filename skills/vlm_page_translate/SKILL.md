# VLM Page Translate Skill

## Purpose

Use one multimodal VLM call to process a small page window and return structured OCR + translation results.

This skill is now job-level batch aware:

- recommended window size: up to `10` pages
- the orchestrator should send the next contiguous page window instead of the whole book
- the output is a document envelope with `page_range` + `pages`

This keeps image context bounded while still allowing the model to see nearby pages together.

---

## Input Contract

### Batch input

Required top-level fields:

- `page_range`
  - `start_page_index`
  - `end_page_index`
  - `page_count`
  - `page_indices`
- `pages`
  - each page item contains:
    - `page_index`
    - `image_bytes`
    - optional `retrieval_context`

Recommended window size:

- `1-10` pages
- default job-level batch size: `10`

The images are sent in the same order as `pages`.

---

## Output Contract

```json
{
  "page_range": {
    "start_page_index": 0,
    "end_page_index": 9,
    "page_count": 10,
    "page_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "label": "0-9"
  },
  "pages": [
    {
      "page_index": 0,
      "page_type": "content",
      "original_image_size": {"width": 1988, "height": 3057},
      "sent_image_size": {"width": 1988, "height": 3057},
      "regions": [
        {
          "region_id": 1,
          "bbox_xyxy": [10, 20, 110, 60],
          "has_text_box": true,
          "color": "black",
          "src_text": "Hello",
          "tgt_text": "你好",
          "utterance_order": 1
        }
      ]
    }
  ]
}
```

Notes:

- `page_range` is the document/window envelope and should be preserved downstream
- every page should include `page_type` = `cover|info|content`
- every page should include `original_image_size` and `sent_image_size`
- `cover` and `info` pages should return an empty `regions` list and skip OCR/translation
- each page keeps its own region ids; region ids only need to be stable within that page
- `bbox_xyxy` is the canonical final bbox format and must be in original-image pixel coordinates
- V1 default is to send the original image without downscaling, so `sent_image_size` usually equals `original_image_size`
- `has_text_box=false` is important for later erase + lettering decisions
- `color` is the preferred text fill color for later lettering; default to `black` if uncertain

---

## Brain / Orchestrator Guidance

- Prefer opening a new VLM batch only when there are no active pages already waiting for revision / erase / lettering.
- Prefer contiguous pages for one batch window.
- Retry single pages separately when only one page fails quality checks.
- Do not resend the whole book just because one page needs a retry.

## Brain Param Spec

```json
{
  "skill_name": "vlm_page_translate",
  "purpose": "detect, OCR, and translate a small page window in one VLM call",
  "when_to_use": [
    "when pipeline_mode=vlm and the next contiguous page window is ready",
    "when the job-level brain wants to open a new 1-10 page batch"
  ],
  "params": {
    "batch_size": {
      "type": "int",
      "min": 1,
      "max": 10,
      "default": 10,
      "effect": "controls how many contiguous pages are sent in one VLM request"
    }
  }
}
```
