# LaMa Eraser Skill

## Purpose

Remove source text from a page image while preserving the artwork underneath.

This skill is exposed as an HTTP service so the orchestrator can run locally on macOS while the heavy inpainting backend runs remotely on a Windows machine with GPU.

Typical flow:

1. Orchestrator provides the original page image.
2. The page may originate from a VLM document window such as `page_range=10-19`.
3. The Windows service first provides remote text-detect artifacts.
4. The client builds a text mask from VLM regions plus remote detect outputs.
5. The HTTP eraser service receives `image + mask + profile`.
6. The service returns an erased page image.
7. The erased image is cached and reused for re-typesetting after translation edits.
8. Pages with art-integrated text (`has_text_box=false`) should still use the embedded-text path, but the current mainline is the hybrid remote-detect mask.

### Client-side mask pipeline

Current default mask chain:

1. Call remote `POST /detect-text-mask` with the source page.
2. Receive:
   - `mask_b64`: raw manga-cleaner detect mask
   - `debug_artifacts.heatmap_b64`: tight text heatmap when available
3. Use VLM regions only to assign/filter connected components per text region.
4. Build a **hybrid mask**:
   - heatmap -> seed
   - raw detect mask -> growth envelope
   - grow the seed inside the envelope with a small kernel
5. Persist debug artifacts:
   - raw detect mask
   - final refined mask
   - overlay
6. Call `POST /inpaint` with the final mask.

Current default strategy:

- `mask_source_mode = manga_auto_hybrid`
- comment: `DC-Nightwing`
- tuned params:
  - `hybrid_seed_safe_expand_px = 5`
  - `hybrid_envelope_safe_expand_px = 40`
  - `hybrid_growth_iterations = 15`
  - `hybrid_growth_kernel_px = 3`

Important constraint:

- VLM bbox should constrain **which components belong to which region**
- VLM bbox should **not** be rasterized as rectangle mask geometry in the final output

This keeps the mask close to real glyph shapes and avoids box-like damage to speech-bubble edges.

The older local heuristic path still exists, but it is now a fallback rather than the preferred route.

### Historical / optional refinement paths

Earlier experiments included heuristic ROI refinement and optional SAM-based refinement. They are kept for fallback/debug, but they are not the preferred default now.

Those paths are still relevant when the current hybrid pipeline:

- clips some glyph strokes
- leaves residue inside the text block
- swallows bubble borders / caption edges

the client can switch `mask_refine_mode` from `heuristic` to `sam_auto` or `sam_required`.

The remote service then follows an `Inpaint-Anything`-style idea:

1. Use the existing heuristic mask as a coarse seed.
2. Convert each text region into SAM prompts:
   - positive points sampled from mask components
   - negative points near the bbox border to protect bubble edges
   - the region bbox as a box prompt
3. Ask SAM for multiple candidate masks.
4. Score the candidates by:
   - coarse-mask recall
   - overlap with a protected edge ring
   - bbox-area ratio
5. Merge the best per-region masks into a refined page mask.

This gives us a practical path toward “erase all text but preserve bubble edges” without trusting raw bbox rectangles.

---

## Input Contract

### HTTP `POST /inpaint`

Required fields:

- `image_b64`: base64-encoded source page image bytes
- `mask_b64`: base64-encoded binary mask bytes (`255` means "erase this pixel")
- `page_index`: current page index

Recommended metadata fields:

- `page_range`
  - the document/window that produced the current text regions
- `batch_id`
  - optional trace id linking this page to the upstream VLM window

Optional fields:

- `backend`: transport-level backend selector
  - recommended value: `lama`
  - reserved for future compatibility or local fallback experiments
- `profile`: inpainting profile
  - `conservative`: small text, low-risk cleanup, preserve texture as much as possible
  - `balanced`: default choice for normal speech/caption text
  - `aggressive`: stronger cleanup for embedded title text / art-integrated text / large serif text

- `mask_source`
  - recommended value from the client metadata:
    - `manga_auto_hybrid`
    - `manga_auto_constrained`
    - `manga_auto_raw`
    - `heuristic`

The client is responsible for deriving the mask from per-region metadata, including:

- `bbox_xyxy` or `bbox_xywh`
- `has_text_box`

### HTTP `POST /refine-mask`

Optional preprocessing endpoint before `/inpaint`.

Required fields:

- `image_b64`
- `coarse_mask_b64`
- `regions`

Per-region fields:

- `region_id`
- `bbox_xywh` or `bbox_xyxy`
- `has_text_box`

Optional tuning fields:

- `positive_points_per_region`
- `negative_points_per_region`
- `edge_protect_px`
- `mask_dilate_px`
- `multimask_output`

Expected response:

- `refined_mask_b64`
- `overlay_b64`
- `backend` such as `sam:vit_h`
- `metrics.region_metrics[]`

---

## Brain / Agent Parameter Guidance

The Brain should not invent arbitrary eraser arguments. It should only choose from supported knobs.

### Main decision knobs

- `mask_source_mode`
  - `manga_auto_hybrid`: current default
  - `manga_auto_constrained`: use constrained raw detect mask without the hybrid growth stage
  - `manga_auto`: use raw detect output directly
  - `heuristic`: old local threshold-based mask path

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
  - only relevant for the heuristic fallback path
  - larger values remove more edge pixels but increase damage risk

- `embedded_box_expand_px`
  - mainly for the heuristic fallback path around `has_text_box=false` regions
  - increase it when the bbox is tight and edge strokes are being missed
  - decrease it when nearby artwork is being swallowed into the mask

- `embedded_complete_mask_close_px`
  - mainly for the heuristic fallback path
  - controls how aggressively the embedded-text path bridges broken glyph strokes
  - increase it when integrated text is fragmented into many small mask islands
  - decrease it when unrelated art details get connected into the same mask

- `hybrid_seed_safe_expand_px`
  - how much the seed assignment may expand around each VLM bbox
  - smaller values keep the seed tighter to visible glyphs
  - larger values help rescue thin edge strokes

- `hybrid_envelope_safe_expand_px`
  - how far the raw detect envelope may extend around each VLM bbox
  - larger values allow more mask growth but raise the risk of swallowing bubble edges

- `hybrid_growth_iterations`
  - how many growth steps the seed may take inside the envelope
  - this is the main aggressiveness knob for residue cleanup

- `hybrid_growth_kernel_px`
  - per-step growth kernel size
  - current tuned default is `3`
  - increasing it is stronger than simply adding a few iterations

- `mask_refine_mode`
  - historical fallback switch, not the current mainline
  - `heuristic`: stay on the old local coarse-mask path
  - `sam_auto`: try remote SAM refinement and fall back to heuristic mask if unavailable
  - `sam_required`: require remote SAM refinement; fail fast if the service/checkpoint is unavailable

- `sam_positive_points_per_region`
  - more points help recover fragmented glyphs
  - start with `4`
  - increase for dense serif text or multi-line narration blocks

- `sam_negative_points_per_region`
  - more negative points strengthen bubble-edge protection
  - start with `8`
  - increase when the refined mask keeps swallowing speech-balloon borders

- `sam_edge_protect_px`
  - width of the edge-protection ring used both in prompting and candidate scoring
  - increase when bubble borders are still being damaged
  - decrease when text lies very close to the bbox boundary and gets under-masked

- `sam_mask_dilate_px`
  - optional small dilation after SAM selection
  - use `0-1` for normal dialogue
  - only push higher when thin glyph edges are repeatedly missed

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

- `lama:manga_cleaner_onnx`
- `lama:simple_lama`
- `lama:saicinpainting_cli`

Recommended local debug artifacts:

- `masks/page_XXXX_coarse.png`
- `masks/page_XXXX_refined.png`
- `masks/page_XXXX_overlay.png`

The orchestrator should persist the returned erased page as a job artifact, typically:

- `data/jobs/<job_id>/erased/page_XXXX.png`

The returned metadata should preserve the single-page identity even when the upstream OCR/translation came from a multi-page VLM window.

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

If neither runner is available, the service should fail loudly so the orchestrator does not silently switch to a poor-quality erase backend.

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
    "mask_source_mode": {
      "type": "enum",
      "values": ["manga_auto_hybrid", "manga_auto_constrained", "manga_auto", "heuristic"],
      "default": "manga_auto_hybrid",
      "effect": "chooses whether erase uses the hybrid remote-detect mask, a simpler constrained/raw detect mask, or the old local heuristic path"
    },
    "profile": {
      "type": "enum",
      "values": ["conservative", "balanced", "aggressive"],
      "default": "balanced",
      "effect": "controls inpainting strength and texture preservation"
    },
    "hybrid_seed_safe_expand_px": {
      "type": "int",
      "min": 0,
      "max": 32,
      "default": 5,
      "effect": "controls how tightly the hybrid seed stays around text components before growth"
    },
    "hybrid_envelope_safe_expand_px": {
      "type": "int",
      "min": 0,
      "max": 128,
      "default": 40,
      "effect": "controls how far the raw detect envelope may extend around assigned components"
    },
    "hybrid_growth_iterations": {
      "type": "int",
      "min": 0,
      "max": 64,
      "default": 15,
      "effect": "main aggressiveness knob for growing the heatmap seed inside the raw detect envelope"
    },
    "hybrid_growth_kernel_px": {
      "type": "int",
      "min": 1,
      "max": 9,
      "default": 3,
      "effect": "per-step growth kernel size; stronger than adding a small number of extra iterations"
    },
    "mask_dilate_px": {
      "type": "int",
      "min": 1,
      "max": 5,
      "default": 2,
      "effect": "larger values erase more edge pixels but increase art damage risk"
    },
    "mask_refine_mode": {
      "type": "enum",
      "values": ["heuristic", "sam_auto", "sam_required"],
      "default": "heuristic",
      "effect": "legacy fallback switch for the optional SAM path; the normal default flow is manga_auto_hybrid without SAM"
    },
    "sam_positive_points_per_region": {
      "type": "int",
      "min": 1,
      "max": 12,
      "default": 4,
      "effect": "number of positive seed points sampled from the coarse text mask"
    },
    "sam_negative_points_per_region": {
      "type": "int",
      "min": 0,
      "max": 16,
      "default": 8,
      "effect": "number of negative protection points near the bbox edge"
    },
    "sam_edge_protect_px": {
      "type": "int",
      "min": 1,
      "max": 64,
      "default": 10,
      "effect": "width of the edge ring used to discourage masks from touching bubble borders"
    },
    "sam_mask_dilate_px": {
      "type": "int",
      "min": 0,
      "max": 8,
      "default": 0,
      "effect": "optional final dilation after SAM candidate selection to recover thin glyph fragments"
    },
    "mask_min_coverage": {
      "type": "float",
      "min": 0.001,
      "max": 0.6,
      "default": 0.01,
      "effect": "lower values accept thinner masks; higher values reject under-covered threshold candidates"
    },
    "mask_max_coverage": {
      "type": "float",
      "min": 0.01,
      "max": 0.9,
      "default": 0.25,
      "effect": "caps oversized mask candidates so artwork is less likely to be swallowed"
    },
    "embedded_box_expand_px": {
      "type": "int",
      "min": 0,
      "max": 16,
      "default": 6,
      "effect": "expands embedded-text ROI before mask extraction to recover tight or clipped bboxes"
    },
    "embedded_complete_mask_close_px": {
      "type": "int",
      "min": 1,
      "max": 15,
      "default": 7,
      "effect": "bridges fragmented embedded-text strokes before LaMa; larger values are stronger but risk merging nearby art"
    }
  }
}
```
