# LaMa Eraser Skill

## Purpose

Remove source text from a page image while preserving the artwork underneath.

This skill is exposed as an HTTP service so the orchestrator can run locally on macOS while the heavy inpainting backend runs remotely on a Windows machine with GPU.

Typical flow:

1. Orchestrator provides the original page image.
2. A text mask is built from VLM/detection regions.
3. The HTTP eraser service receives `image + mask + profile`.
4. The service returns an erased page image.
5. The erased image is cached and reused for re-typesetting after translation edits.
6. Pages with art-integrated text (`has_text_box=false`) should use the embedded-text mask path before calling LaMa.

---

## Input Contract

### HTTP `POST /inpaint`

Required fields:

- `image_b64`: base64-encoded source page image bytes
- `mask_b64`: base64-encoded binary mask bytes (`255` means "erase this pixel")

Optional fields:

- `backend`: transport-level backend selector
  - recommended value: `lama`
  - reserved for future compatibility or local fallback experiments
- `profile`: inpainting profile
  - `conservative`: small text, low-risk cleanup, preserve texture as much as possible
  - `balanced`: default choice for normal speech/caption text
  - `aggressive`: stronger cleanup for embedded title text / art-integrated text / large serif text

The client is responsible for deriving the mask from per-region metadata, including:

- `bbox_xywh`
- `has_text_box`

---

## Brain / Agent Parameter Guidance

The Brain should not invent arbitrary eraser arguments. It should only choose from supported knobs.

### Main decision knobs

- `profile`
  - choose `balanced` by default
  - retry with `aggressive` when:
    - text is embedded directly into art
    - first erase leaves visible source glyph remnants
    - text strokes are heavy / serif / decorative
  - retry with `conservative` when:
    - artwork is very dense
    - first erase damages nearby line art or shading

- `mask_dilate_px`
  - this is generated on the client side before the HTTP call
  - larger values remove more edge pixels but increase damage risk
  - suggested policy:
    - `2` default
    - `3-4` for thick title text
    - `1` for delicate line art

- `has_text_box`
  - `true`: standard bubble/caption cleanup path
  - `false`: embedded-text path with stronger mask completion and usually `aggressive` profile
  - if absent: treat as unknown and fall back to the standard path

- `mask_min_coverage` / `mask_max_coverage`
  - client-side heuristics for selecting the ROI threshold result
  - use defaults unless the page repeatedly under-masks or over-masks text

---

## Expected Output

- `image_b64`: base64-encoded erased page image
- `backend`: backend actually used
- `profile`: inpainting profile actually used

Typical `backend` values returned by the server:

- `lama:simple_lama`
- `lama:saicinpainting_cli`
- `opencv`
- `opencv:auto_fallback`

Recommended local debug artifacts:

- `masks/page_XXXX_coarse.png`
- `masks/page_XXXX_refined.png`
- `masks/page_XXXX_overlay.png`

The orchestrator should persist the returned erased page as a job artifact, typically:

- `data/jobs/<job_id>/erased/page_XXXX.png`

---

## Retry Policy

Recommended order:

1. `balanced`
2. `aggressive` if text residue remains
3. `conservative` if art damage is observed
4. ask user if erase quality is still unacceptable

Do not rerun VLM just because erase quality is poor if the text regions themselves are already correct.

---

## Notes

- This skill should be independent from typesetting.
- Translation edits should reuse the cached erased page whenever possible.
- Long term, this HTTP service is the best place to host a true LaMa model on Windows + CUDA.

### Recommended Windows deployment

Two supported ways to run real LaMa:

1. `simple_lama`
   - easiest setup
   - install the optional Python package dependency
   - set `LAMA_RUNNER=simple_lama`

2. `saicinpainting_cli`
   - closer to the official LaMa repo
   - clone the LaMa repository and download the checkpoint directory
   - set:
     - `LAMA_RUNNER=saicinpainting_cli`
     - `LAMA_REPO_DIR=<path to lama repo>`
     - `LAMA_MODEL_DIR=<path to big-lama checkpoint dir>`
     - optional: `LAMA_DEVICE=cuda`
     - optional: `LAMA_PYTHON_EXE=<python in that env>`

When `LAMA_RUNNER=auto`, the server tries:

1. official `saicinpainting_cli` if repo + model dirs are configured
2. `simple_lama` if the package is installed
3. OpenCV fallback for local development only

## Brain Param Spec

```json
{
  "skill_name": "lama_eraser",
  "purpose": "erase source text while preserving artwork",
  "when_to_use": [
    "after translation is acceptable and before typesetting",
    "retry with a different profile when residue or art damage is observed"
  ],
  "params": {
    "profile": {
      "type": "enum",
      "values": ["conservative", "balanced", "aggressive"],
      "default": "balanced",
      "effect": "controls inpainting strength and texture preservation"
    },
    "mask_dilate_px": {
      "type": "int",
      "min": 1,
      "max": 5,
      "default": 2,
      "effect": "larger values erase more edge pixels but increase art damage risk"
    }
  }
}
```
