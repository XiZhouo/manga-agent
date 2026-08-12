# Windows SAM Mask Refine Service Spec

This file is the only new spec the Windows-side agent needs in order to extend the existing FastAPI LaMa service with a SAM-based mask refinement stage.

Read this as an implementation document, not as a high-level idea note.

The goal is very specific:

- macOS already builds a heuristic text mask from bbox + threshold refinement
- that heuristic mask is often close, but still has two failure modes:
  - it can damage speech-bubble borders / caption-box borders
  - it can miss thin glyph strokes, leaving residual source text
- Windows should expose a new synchronous endpoint that uses SAM to refine the incoming coarse mask
- macOS will call this new endpoint before calling `/inpaint`

This spec is for the remote Windows service only.

Non-goals:

- no OCR
- no VLM
- no bbox generation
- no UI click tool
- no async queue
- no websocket
- no background job polling API

The existing `/inpaint` endpoint must keep working.

---

## 1. What problem this solves

Current macOS flow:

1. macOS has page image
2. macOS has region bbox list
3. macOS builds a heuristic mask
4. macOS sends `image + mask` to Windows `/inpaint`
5. Windows runs LaMa

Observed problems:

- coarse mask may include bubble edges
- coarse mask may fail to include all text strokes

New target flow:

1. macOS builds heuristic coarse mask exactly as before
2. macOS sends `image + coarse_mask + region metadata` to Windows `/refine-mask`
3. Windows runs SAM and returns a refined binary mask
4. macOS uses the refined mask as the final erase mask
5. macOS sends `image + refined_mask` to Windows `/inpaint`

In other words:

- `/refine-mask` improves mask quality
- `/inpaint` still performs actual image removal

---

## 2. Required behavior summary

The Windows service must now support four HTTP endpoints:

- `GET /health`
- `GET /capabilities`
- `POST /refine-mask`
- `POST /inpaint`

Rules:

- all endpoints return JSON
- `/refine-mask` is synchronous
- `/inpaint` remains synchronous
- the service must fail loudly when SAM is required but unavailable
- the service must not silently invent bbox or alter region count

---

## 3. Runtime assumptions

Expected deployment:

- service runs on Windows with GPU
- SAM and LaMa both run on Windows
- macOS calls Windows over LAN

Typical stack:

- Python 3.10 or 3.11
- FastAPI
- Uvicorn
- numpy
- pillow
- opencv-python
- torch
- segment-anything
- LaMa backend already chosen by your current Windows setup

Supported LaMa backends remain:

- `inpaint_anything_lama`
- `simple_lama`
- `saicinpainting_cli`

The exact LaMa backend is independent from the SAM refinement endpoint.

---

## 4. Environment variables

Keep all existing LaMa env vars.

Add these new SAM env vars:

### Required for SAM

- `LAMA_SAM_ENABLED`
  - allowed values:
    - `off`
    - `auto`
    - `on`
  - recommended during rollout: `on`

- `LAMA_SAM_CHECKPOINT`
  - absolute path to the SAM checkpoint file
  - example:
    - `D:\models\sam\sam_vit_h_4b8939.pth`

### Optional

- `LAMA_SAM_MODEL_TYPE`
  - allowed values depend on checkpoint
  - common values:
    - `vit_h`
    - `vit_l`
    - `vit_b`
  - default: `vit_h`

- `LAMA_SAM_DEVICE`
  - recommended:
    - `cuda`
  - fallback:
    - `cpu`

Example PowerShell setup:

```powershell
$env:LAMA_SAM_ENABLED="on"
$env:LAMA_SAM_CHECKPOINT="D:\models\sam\sam_vit_h_4b8939.pth"
$env:LAMA_SAM_MODEL_TYPE="vit_h"
$env:LAMA_SAM_DEVICE="cuda"
```

Important:

- if `LAMA_SAM_ENABLED=on` and the checkpoint is missing, `/refine-mask` must fail
- if `LAMA_SAM_ENABLED=auto` and the checkpoint is missing, `/refine-mask` may still fail, but `GET /capabilities` must clearly report why SAM is unavailable

---

## 5. `GET /health`

Purpose:

- quick process liveness check

Response:

```json
{
  "ok": true,
  "name": "LamaEraseServer",
  "version": "0.3.0",
  "runner": "inpaint_anything_lama",
  "sam": "available"
}
```

Required fields:

- `ok`
- `name`
- `version`
- `runner`
- `sam`

Allowed `sam` values:

- `available`
- `disabled`
- `missing_checkpoint`
- `missing_dependency`

Notes:

- `runner` refers to LaMa/inpaint backend availability
- `sam` refers to SAM refine availability

---

## 6. `GET /capabilities`

Purpose:

- tell macOS what this server can currently do

Response example:

```json
{
  "ok": true,
  "backend": "lama",
  "runner": "inpaint_anything_lama",
  "profiles": ["conservative", "balanced", "aggressive"],
  "mask_refine_modes": ["heuristic", "sam_auto", "sam_required"],
  "sam": {
    "status": "available",
    "model_type": "vit_h",
    "device": "cuda"
  },
  "env": {
    "LAMA_RUNNER": "auto",
    "LAMA_REPO_DIR": true,
    "LAMA_MODEL_DIR": true,
    "LAMA_DEVICE": "cuda",
    "LAMA_SAM_ENABLED": "on",
    "LAMA_SAM_CHECKPOINT": true,
    "LAMA_SAM_MODEL_TYPE": "vit_h",
    "LAMA_SAM_DEVICE": "cuda"
  }
}
```

Required additions compared with the old service:

- `mask_refine_modes`
- `sam.status`
- `sam.model_type`
- `sam.device`
- `env.LAMA_SAM_ENABLED`
- `env.LAMA_SAM_CHECKPOINT`
- `env.LAMA_SAM_MODEL_TYPE`
- `env.LAMA_SAM_DEVICE`

---

## 7. New endpoint: `POST /refine-mask`

### 7.1 Purpose

Refine an incoming heuristic text mask using SAM.

The server must:

1. decode the page image
2. decode the coarse mask
3. read region bbox list
4. generate SAM prompts per region
5. ask SAM for candidate masks
6. score the candidates
7. merge the chosen region masks into one final page mask
8. return:
   - the refined mask
   - an optional debug overlay
   - scoring metrics

### 7.2 Request JSON

```json
{
  "image_b64": "<base64 source page image>",
  "coarse_mask_b64": "<base64 binary mask png>",
  "regions": [
    {
      "region_id": 1,
      "bbox_xywh": [136, 166, 341, 130],
      "has_text_box": true
    },
    {
      "region_id": 2,
      "bbox_xywh": [786, 126, 400, 170],
      "has_text_box": true
    }
  ],
  "profile": "balanced",
  "positive_points_per_region": 4,
  "negative_points_per_region": 8,
  "edge_protect_px": 10,
  "multimask_output": true,
  "mask_dilate_px": 0
}
```

### 7.3 Request fields

Required:

- `image_b64`
- `coarse_mask_b64`
- `regions`

Optional:

- `profile`
- `positive_points_per_region`
- `negative_points_per_region`
- `edge_protect_px`
- `multimask_output`
- `mask_dilate_px`

Per-region object:

- `region_id`: required integer
- `bbox_xywh`: optional list of 4 ints
- `bbox_xyxy`: optional list of 4 ints
- `has_text_box`: optional bool

Rules:

- at least one of `bbox_xywh` or `bbox_xyxy` must be present
- `bbox_xywh` means `[x, y, w, h]`
- `bbox_xyxy` means `[x1, y1, x2, y2]`
- if both are present, prefer `bbox_xyxy`
- `has_text_box=true` means bubble/caption text
- `has_text_box=false` means text embedded in artwork

### 7.4 Mask semantics

Input coarse mask:

- `0` = keep
- `255` = text candidate / erase candidate

Returned refined mask:

- `0` = keep
- `255` = erase

### 7.5 Success response

```json
{
  "ok": true,
  "backend": "sam:vit_h",
  "refined_mask_b64": "<base64 png mask>",
  "overlay_b64": "<base64 debug overlay png>",
  "metrics": {
    "regions": 5,
    "mask_pixels": 182345,
    "mask_coverage": 0.030112,
    "sam_model_type": "vit_h",
    "sam_device": "cuda",
    "region_metrics": [
      {
        "region_id": 1,
        "bbox_xyxy": [136, 166, 477, 296],
        "has_text_box": true,
        "positive_points": [[220, 192], [302, 215], [410, 267]],
        "negative_points": [[146, 176], [306, 176], [466, 176]],
        "selected_index": 2,
        "score": 2.744,
        "coarse_recall": 0.94,
        "area_ratio": 0.18,
        "edge_overlap": 0.03,
        "oversize_penalty": 0.0,
        "recall_penalty": 0.0,
        "candidate_pixels": 7021,
        "sam_score": 0.978
      }
    ]
  }
}
```

Required success fields:

- `ok`
- `backend`
- `refined_mask_b64`
- `metrics`

Optional but strongly recommended:

- `overlay_b64`

### 7.6 Failure behavior

Use JSON failure responses.

Examples:

```json
{
  "ok": false,
  "error": {
    "code": "sam_unavailable",
    "message": "SAM checkpoint is missing or dependencies are unavailable."
  }
}
```

Stable failure codes to support:

- `sam_unavailable`
- `bad_base64`
- `bad_image`
- `bad_mask`
- `empty_regions`
- `invalid_bbox`
- `sam_predict_failed`
- `encode_mask_failed`

If FastAPI raises an `HTTPException`, convert the response to normal JSON structure if practical.
If not, at minimum keep a compact JSON body, not HTML.

---

## 8. Exact SAM refinement strategy

This section is the most important part of the spec.

The Windows-side agent should not invent a different strategy unless blocked.

### 8.1 Predictor model

Use:

- `segment_anything`
- `SamPredictor`
- `sam_model_registry[model_type](checkpoint=...)`

Then:

- move model to `LAMA_SAM_DEVICE`
- call `predictor.set_image(image_rgb)` once per request

### 8.2 Region bbox normalization

For every region:

1. convert `bbox_xywh` to `bbox_xyxy` if needed
2. clamp bbox to image bounds
3. skip invalid bboxes

Expected normalization:

- `x1 >= 0`
- `y1 >= 0`
- `x2 > x1`
- `y2 > y1`
- `x2 <= image width`
- `y2 <= image height`

### 8.3 Positive point generation

Use the incoming coarse mask as the seed.

Per region:

1. crop the coarse mask to the bbox ROI
2. find connected components in that ROI
3. take up to `positive_points_per_region` component centroids
4. if centroids are fewer than requested:
   - sample more points from nonzero mask pixels
5. if still empty:
   - use bbox center as a fallback positive point

Important:

- positive points should prefer actual text-mask pixels
- do not place positive points outside the bbox

### 8.4 Negative point generation

Negative points exist to protect bubble borders and nearby art.

For boxed text (`has_text_box=true`):

- place negative points near the bbox border
- recommended 8-point layout:
  - top-left inset
  - top-center inset
  - top-right inset
  - mid-left inset
  - mid-right inset
  - bottom-left inset
  - bottom-center inset
  - bottom-right inset

Inset size:

- `edge_protect_px`
- but clamp so it fits small boxes

For embedded text (`has_text_box=false`):

- use fewer negative points
- still allow some border protection
- do not over-constrain, because text may blend into art

### 8.5 SAM call

For each region, call:

- `predictor.predict(...)`

Inputs:

- `point_coords`
- `point_labels`
- `box`
- `multimask_output=true`

Expected result:

- multiple candidate masks
- one SAM confidence score per candidate

### 8.6 Candidate scoring

Do not blindly take SAM candidate `0`.

Score every candidate using these signals:

#### A. Coarse mask recall

Within the region bbox:

- `coarse_recall = overlap(candidate, coarse_mask) / area(coarse_mask)`

Meaning:

- the refined mask should still cover most of the original text seed

#### B. Edge overlap penalty

Build a border-protection ring inside the bbox:

- start with full bbox area
- carve out an inner rectangle inset by `edge_protect_px`
- remaining ring is the edge-protection zone

Then:

- `edge_overlap = overlap(candidate, edge_ring) / area(edge_ring)`

Meaning:

- candidate touching the border ring too much is suspicious for bubble damage

#### C. Area ratio

- `area_ratio = candidate_pixels / bbox_area`

Use different targets for boxed vs embedded text:

- boxed text target area ratio:
  - around `0.16`
- embedded text target area ratio:
  - around `0.28`

Use hard oversize thresholds:

- boxed max area ratio:
  - `0.45`
- embedded max area ratio:
  - `0.70`

#### D. Recall penalty

If coarse recall is below a threshold, penalize strongly.

Recommended minimum:

- `0.72`

### 8.7 Suggested scoring formula

Use a weighted score similar to:

```text
score =
  (4.5 * coarse_recall)
  - (2.8 * edge_overlap)
  - (2.2 * abs(area_ratio - target_area))
  - (4.0 * oversize_penalty)
  - (6.0 * recall_penalty)
  + (0.15 * sam_score)
```

Where:

- `oversize_penalty = max(0, area_ratio - max_area_ratio)`
- `recall_penalty = max(0, min_coarse_recall - coarse_recall)`

Exact numbers may be tuned slightly, but keep the same intent:

- prioritize recall
- punish edge damage
- punish giant masks

### 8.8 Final page mask assembly

After choosing one candidate per region:

1. optionally apply a very small dilation if `mask_dilate_px > 0`
2. merge all region masks using pixelwise max
3. return the merged binary mask

Do not run large dilation by default.

Reason:

- we are trying to reduce border damage, not expand aggressively

---

## 9. Overlay image requirements

Return a debug overlay image via `overlay_b64`.

Recommended contents:

- original page image as background
- refined mask shown in semi-transparent green
- bbox outlines in yellow
- positive points in green
- negative points in red

This overlay is very important for debugging.

The macOS side will inspect it to answer:

- are positive points landing on text?
- are negative points guarding bubble borders?
- is the refined mask still too large?

---

## 10. `/inpaint` endpoint remains required

Keep `/inpaint` behavior unchanged.

Current contract:

- input:
  - `image_b64`
  - `mask_b64`
  - `backend`
  - `profile`
- output:
  - erased `image_b64`
  - actual backend used
  - profile used

Important:

- `/inpaint` must not regenerate or reinterpret the mask
- `/refine-mask` is responsible for mask optimization
- `/inpaint` just uses the final mask it receives

---

## 11. Capabilities and rollout expectations

The macOS client will use three logical mask refine modes:

- `heuristic`
  - no SAM call
  - old behavior

- `sam_auto`
  - try `/refine-mask`
  - if it fails, fall back to heuristic mask

- `sam_required`
  - require `/refine-mask`
  - if it fails, whole erase step fails

So the Windows implementation must support:

- a stable `/refine-mask`
- clear capability reporting
- clean failure messages

---

## 12. Suggested internal implementation structure

The Windows agent can choose its own files, but a clean structure would be:

```text
skills/
  lama_erase_server/
    server.py
    sam_refine.py
    lama_runner.py
    image_io.py
```

Suggested responsibilities:

- `server.py`
  - FastAPI routes
  - request/response schemas

- `sam_refine.py`
  - checkpoint loading
  - prompt generation
  - candidate scoring
  - overlay rendering

- `lama_runner.py`
  - existing inpaint-anything / LaMa calls

- `image_io.py`
  - base64 encode/decode
  - PNG mask encode/decode

This is only a suggestion, not a hard requirement.

---

## 13. Recommended dependency list

Minimum:

```powershell
pip install fastapi uvicorn pillow numpy opencv-python requests
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
```

If internet install is inconvenient, the Windows agent can use a local wheel / local clone approach.

LaMa-related dependencies remain whatever your current inpaint backend already needs.

---

## 14. Startup example

Example PowerShell startup:

```powershell
$env:LAMA_RUNNER="auto"
$env:LAMA_DEVICE="cuda"
$env:LAMA_SAM_ENABLED="on"
$env:LAMA_SAM_CHECKPOINT="D:\models\sam\sam_vit_h_4b8939.pth"
$env:LAMA_SAM_MODEL_TYPE="vit_h"
$env:LAMA_SAM_DEVICE="cuda"
python -m uvicorn skills.lama_erase_server.server:app --host 0.0.0.0 --port 8000
```

After startup, expected checks:

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/capabilities
```

The second call should report:

- `sam.status = available`

---

## 15. Manual test case the Windows agent should run

The Windows agent should test with one real page image and one coarse mask.

### Test A: `/refine-mask`

Input:

- a real comic page
- a real coarse mask
- 5 text regions

Verify:

- response status `200`
- `refined_mask_b64` decodes correctly
- overlay image decodes correctly
- `metrics.region_metrics` length matches usable region count
- refined mask is not empty
- refined mask is not almost full-page

### Test B: refined mask + `/inpaint`

Take the refined mask from Test A and send it to `/inpaint`.

Verify:

- response status `200`
- output image decodes correctly
- output image dimensions equal input dimensions

### Test C: capability failure

Set:

- `LAMA_SAM_ENABLED=on`
- invalid or missing checkpoint

Verify:

- `/health` reports unavailable SAM state
- `/capabilities` reports `sam.status != available`
- `/refine-mask` fails with JSON error

---

## 16. Things the Windows agent must not do

- do not make `/refine-mask` asynchronous
- do not require a GUI click workflow
- do not replace the incoming coarse mask with a pure bbox rectangle
- do not remove `/inpaint`
- do not hide SAM failure behind silent fallback when the caller requested required behavior
- do not aggressively dilate by default
- do not return HTML error pages

---

## 17. What to send back after implementation

After the Windows agent finishes, it should report:

- what files it added or modified
- which SAM model type it used
- exact checkpoint path it expects
- exact startup command
- result of:
  - `GET /health`
  - `GET /capabilities`
- one successful `/refine-mask` smoke test summary
- one successful `/inpaint` smoke test summary using the refined mask

That is enough for macOS to begin live integration tests.

