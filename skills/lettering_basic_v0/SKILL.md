# Lettering Skill

## Purpose

Typeset translated text back onto an already-erased page image.

This skill should never erase source text itself. Erasing is handled by the dedicated eraser skill so that:

- translation revisions can reuse cached erased pages
- retries can target typography only
- orchestrator state stays modular

---

## Input Contract

Required fields:

- `image_bytes`: erased page image bytes
- `page_index`: current page index
- `regions`: list of translated regions
  - `region_id`
  - `bbox_xywh` / `bbox_xyxy` as fallback geometry
  - `has_text_box`
  - `color`
  - `translated_text`

Preferred shape-aware fields:

- `placement_lines`
  - generated from the final erase mask, not from the raw VLM bbox
  - each line describes one writable band inside the bubble
  - recommended per-line fields:
    - `x_left`
    - `x_right`
    - `y_top`
    - `y_bottom`
    - `x_center`
    - `y_center`
    - `usable_width`
    - `usable_height`

When `placement_lines` is present, lettering should prefer it over bbox-based rectangle fitting.

Recommended metadata fields:

- `page_range`
  - original document/window range that produced the translation text
- `batch_id`
  - optional upstream batch trace id

---

## Main Parameters

These are the knobs the Brain may choose from when retrying:

- `render_mode`
  - `bubble_fit`: default for `has_text_box=true`
  - `art_embed`: default for `has_text_box=false`; tighter padding, larger font search, stronger outline
  - the Brain should normally not override this unless page review says the auto-selected mode is wrong

- `placement_source`
  - `placement_lines`: preferred; use mask-derived writable bands
  - `bbox`: fallback only when region-level placement lines are missing

- `padding_ratio`
  - default `0.08`
  - increase when text touches bubble edges
  - decrease when text is too compressed

- `min_font_size`
  - lower bound for fitting text

- `max_font_size_ratio`
  - controls initial font size search range relative to bbox height

- `line_spacing`
  - increase when lines feel cramped
  - decrease when overflow happens only vertically

- `stroke_width`
  - larger values improve readability on noisy backgrounds
  - smaller values preserve a lighter manga look

For art-integrated text (`has_text_box=false`), the skill uses a different default style:

- smaller padding
- larger font ceiling
- white fill with black outline to preserve over-art readability

Color guidance:

- if `color` is present, lettering should prefer that fill color instead of the mode default
- if `color` is missing, fall back to the existing mode default
- the outline/stroke should be chosen for contrast automatically

---

## Output

- `rendered_image_bytes`
- `results`
  - `render_ok`
  - `overflow`
  - `render_params`

The orchestrator should persist the rendered page separately from the erased page, typically:

- `data/jobs/<job_id>/rendered/page_XXXX.png`

Even when the source translation arrived from a 10-page VLM window, lettering still runs per page and should keep page-local output files.

---

## Retry Guidance

When typesetting quality is poor:

- if text overflows vertically: lower `line_spacing`, lower font size ceiling, or split text more aggressively
- if text overflows horizontally: increase wrapping or reduce font size
- if readability is poor on textured backgrounds: increase `stroke_width`
- if text looks too cramped inside large bubbles: increase `padding_ratio` slightly and retry
- if VLM marked the region as `has_text_box=false`, prefer `art_embed` retry behavior instead of bubble-style tuning

Do not rerun the eraser skill for typography-only issues unless the erased background is clearly wrong.

## Brain Param Spec

```json
{
  "skill_name": "lettering",
  "purpose": "typeset translated text onto an erased page",
  "when_to_use": [
    "after erase has succeeded",
    "retry for overflow, readability, or style mismatch without rerunning erase"
  ],
  "params": {
    "render_mode": {
      "type": "enum",
      "values": ["bubble_fit", "art_embed"],
      "default": "bubble_fit",
      "effect": "chooses bubble-oriented fitting or art-integrated rendering style"
    },
    "padding_ratio": {
      "type": "float",
      "min": 0.01,
      "max": 0.2,
      "default": 0.08,
      "effect": "inner padding inside the target region"
    },
    "line_spacing": {
      "type": "float",
      "min": 0.8,
      "max": 1.5,
      "default": 1.15,
      "effect": "controls vertical density of wrapped lines"
    },
    "stroke_width": {
      "type": "int",
      "min": 0,
      "max": 6,
      "default": 2,
      "effect": "controls text outline thickness for readability"
    },
    "max_font_size_ratio": {
      "type": "float",
      "min": 0.3,
      "max": 0.9,
      "default": 0.6,
      "effect": "upper bound of font search relative to box height"
    }
  }
}
```
