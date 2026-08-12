# Windows Manga-Cleaner Service Spec

This file is the implementation spec for the Windows-side agent.

Goal:

- adapt the `manga-cleaner` backend into an HTTP service
- keep the macOS agent unchanged as much as possible
- reuse the upstream default processing chain and default parameters first
- expose the cleaner as a mask-detect + inpaint service that can run side-by-side with the current LaMa service during rollout

Important:

- do not rebuild the upstream desktop UI
- do not copy only the "idea" and then heavily redesign the pipeline on day one
- start from the upstream defaults, then add only the minimum service wrapper and observability needed by our agent
- preserve the original upstream order of operations as much as possible

Pure-service principle:

- the first implementation should feel like "the original project without the UI"
- if the original project auto-detects text first and then cleans, the service should do the same
- only add compatibility wrappers and debug output; avoid changing thresholds or morphology defaults in v1

License / attribution requirement:

- upstream author has granted reuse permission to this project
- when packaging or publishing, preserve attribution to:
  - `NeTRuNNeRGLiTCH/manga-cleaner`
- do not remove upstream notices
- do not use the resulting project for commercial purposes

---

## 1. Deployment strategy

Do **not** replace the current service immediately.

Recommended rollout:

1. keep the current Windows service running if convenient
2. clone `manga-cleaner` on Windows
3. build a new service wrapper around its backend code
4. expose the new backend under the same FastAPI process or under a new port
5. test against the same page images and masks
6. switch macOS to the new backend only after parity is confirmed

Reason:

- current service is the fallback
- rollback becomes trivial
- debugging is easier if both implementations can be compared on the same request

Recommended repository layout on Windows:

- option A: clone upstream repo as a sibling project and build a small service adapter next to it
- option B: clone upstream repo and add a `service/` folder inside that clone for FastAPI wrappers

Recommended choice for speed:

- option B, because the Windows-side agent can directly import upstream modules without repackaging first

Example:

```text
D:\ai\manga-cleaner\
  main.py
  requirements.txt
  models\
  src\
  service\
    app.py
    adapters.py
    schemas.py
```

---

## 2. High-level architecture

The macOS side already does:

- CBZ/CBR extraction
- page batching
- VLM OCR + translation
- revision
- local mask generation from region metadata
- future quality checks and retries

The Windows side should only do erase-related work.

New target backend split:

- macOS:
  - decide whether to use:
    - local VLM-derived mask
    - remote `manga-cleaner` auto-scan mask
    - a merged mask
  - decide erase profile
  - call remote service
- Windows:
  - optionally run upstream auto-scan text detection
  - optionally receive an external mask
  - run upstream `manga-cleaner` style cleaning
  - return mask / erased image / debug metadata

Phase 1 scope:

- reuse upstream text scan + inpainting chain
- expose upstream auto-scan as a separate endpoint
- keep external-mask inpaint support for comparison and fusion
- keep the service behavior as close to the upstream app as possible

Phase 2 scope:

- only after evaluating the pure upstream flow, consider adding:
  - mask fusion policies
  - extra safety constraints for bubble edges
  - custom profile tuning for embedded text

---

## 3. What to reuse from upstream first

The Windows-side implementation should reuse upstream logic in this order:

1. model loading pattern
   - ONNX Runtime session management
   - CUDA first, CPU fallback
2. image preprocessing
   - padding rules
   - shape alignment for model input
3. text auto-scan flow
   - OCR heatmap generation
   - thresholding
   - morphology defaults
4. local ROI cleaning flow
   - connected components over mask
   - clean per blob / per local tile instead of one whole-page pass
5. default thresholds and morphological defaults
   - keep upstream defaults initially
6. output composition
   - paste cleaned local results back into the full page

Do **not** change these on the first pass unless needed to make the HTTP service work.

Reason:

- we want to validate the upstream workflow itself before tuning
- if quality is bad, we need to know whether the issue is:
  - mask quality
  - upstream defaults
  - service adaptation bug

---

## 4. What not to reuse

Do not port:

- desktop UI
- keyboard shortcut handling
- Photoshop integration
- manual brush / lasso / rectangle interaction layers
- export dialogs
- project/session UI state

These features belong to the studio app, not to our agent backend.

---

## 5. Service compatibility goals

The new service should preserve the existing remote contract as much as possible.

At minimum, support:

- `GET /health`
- `GET /capabilities`
- `POST /detect-text-mask`
- `POST /inpaint`

Optional:

- `POST /inpaint-from-detect`
  - convenience endpoint for the pure upstream flow:
    - detect mask
    - inpaint
    - return both results

The easiest integration path is:

- keep `/inpaint` request schema largely compatible with the current service
- add a new detect step before inpaint on macOS
- keep the old backend available in parallel

---

## 6. Required endpoints

### `GET /health`

Purpose:

- liveness check for macOS

Response example:

```json
{
  "ok": true,
  "name": "InpaintingService",
  "version": "0.2.0",
  "runner": "manga_cleaner_onnx"
}
```

Required fields:

- `ok`
- `name`
- `version`
- `runner`

Allowed `runner` examples:

- `manga_cleaner_onnx`
- `inpaint_anything_lama`
- `simple_lama`
- `saicinpainting_cli`
- `unavailable`

---

### `GET /capabilities`

Purpose:

- tell macOS what backends and options are available

Response example:

```json
{
  "ok": true,
  "backend": "lama",
  "runner": "manga_cleaner_onnx",
  "profiles": ["conservative", "balanced", "aggressive"],
  "supports": {
    "external_mask": true,
    "roi_cleaning": true,
    "ocr_auto_mask": true,
    "sam_refine": false
  },
  "limits": {
    "max_image_side": 0,
    "request_timeout_sec": 180
  }
}
```

Notes:

- `backend` can remain `"lama"` for compatibility with the macOS caller
- `runner` should identify the actual implementation
- `ocr_auto_mask` should be `true` if the upstream detect chain is wired
- `sam_refine` should reflect upstream reality, not our previous service assumptions

---

### `POST /detect-text-mask`

Purpose:

- run the upstream `manga-cleaner` text auto-scan chain
- return the mask produced by the upstream defaults

This endpoint is the service equivalent of the upstream UI action:

- `Load -> Auto-Scan`

Request JSON:

```json
{
  "request_id": "job123-page0002-detect1",
  "image_b64": "<base64 image bytes>",
  "profile": "balanced",
  "timeout_sec": 180,
  "debug": true
}
```

Required request fields:

- `image_b64`

Optional request fields:

- `request_id`
- `profile`
- `timeout_sec`
- `debug`

Success response:

```json
{
  "ok": true,
  "request_id": "job123-page0002-detect1",
  "mask_b64": "<base64 binary mask png>",
  "backend": "mask:manga_cleaner_ocr",
  "profile": "balanced",
  "latency_ms": 2140,
  "warnings": [],
  "metrics": {
    "mask_coverage": 0.018,
    "component_count": 7
  },
  "debug_artifacts": {
    "heatmap_b64": "<optional>",
    "overlay_b64": "<optional>"
  }
}
```

Behavior requirements:

- use upstream defaults first
- do not replace upstream threshold or dilation with custom values in v1
- if the upstream code outputs a heatmap before binarization, return it in debug mode
- the returned `mask_b64` must be a binary PNG with:
  - `0 = keep`
  - `255 = erase`

This endpoint is the most important part of the adaptation.

---

### `POST /inpaint`

Purpose:

- perform inpainting using either:
  - an external mask from macOS
  - or a mask produced earlier by `/detect-text-mask`

This endpoint is the service equivalent of the upstream UI action:

- `AI Clean`

Request JSON:

```json
{
  "request_id": "job123-page0002-try1",
  "image_b64": "<base64 image bytes>",
  "mask_b64": "<base64 mask png bytes>",
  "backend": "lama",
  "profile": "balanced",
  "timeout_sec": 180,
  "debug": true,
  "mask_source": "external"
}
```

Required request fields:

- `image_b64`
- `mask_b64`
- `backend`
- `profile`

Optional request fields:

- `request_id`
- `timeout_sec`
- `debug`
- `mask_source`
  - recommended values:
    - `external`
    - `manga_cleaner_detect`
    - `merged`

Mask semantics:

- `0` = keep
- `255` = erase

Success response:

```json
{
  "ok": true,
  "request_id": "job123-page0002-try1",
  "image_b64": "<base64 erased png bytes>",
  "backend": "lama:manga_cleaner_onnx",
  "profile": "balanced",
  "latency_ms": 6123,
  "warnings": [],
  "metrics": {
    "mask_coverage": 0.021,
    "component_count": 5,
    "roi_count": 5
  },
  "debug_artifacts": {
    "mask_used_b64": "<optional>",
    "overlay_b64": "<optional>"
  }
}
```

Failure response:

```json
{
  "ok": false,
  "request_id": "job123-page0002-try1",
  "error": {
    "code": "backend_error",
    "message": "manga_cleaner backend failed",
    "detail": "onnx runtime session error"
  },
  "backend": "lama:manga_cleaner_onnx",
  "profile": "balanced",
  "latency_ms": 884,
  "warnings": []
}
```

Important:

- always return JSON
- keep HTTP 200 for logical success/failure if that matches the current service style
- only use 4xx/5xx for malformed requests or unexpected framework failures

---

## 7. Profile mapping

The macOS side already reasons in:

- `conservative`
- `balanced`
- `aggressive`

The Windows service should map these logical profiles onto upstream cleaner parameters.

First-pass rule:

- keep upstream defaults for `balanced`
- if the upstream project has no explicit profile concept, emulate profiles as thin wrappers only
- derive `conservative` and `aggressive` as narrow wrappers around those defaults

Recommended initial mapping:

- `conservative`
  - reduce mask growth if the backend applies any expansion internally
  - prefer smaller ROI padding where safe
  - intended for pages with delicate bubble edges or line art
- `balanced`
  - exact upstream defaults
- `aggressive`
  - allow slightly larger effective context / padding
  - keep all other upstream defaults unless there is a clear reason to change them

Do not invent many profile knobs in v1.

The main point is to preserve upstream behavior, not to redesign it.

---

## 8. Request handling chain

The service should process requests in this order.

Pure upstream path:

1. decode `image_b64`
2. run upstream auto-scan logic
3. return the mask
4. macOS inspects or stores the mask
5. macOS sends `image + mask` to `/inpaint`
6. service runs upstream cleaning

External-mask comparison path:

1. decode `image_b64`
2. decode `mask_b64`
3. verify image and mask sizes match
4. normalize mask into a binary `uint8` image
5. compute light diagnostics:
   - image size
   - mask coverage
   - connected component count
6. route to the `manga_cleaner_onnx` adapter
7. adapter runs upstream cleaning chain using the provided mask
8. re-encode output as PNG base64
9. return JSON with metrics

Important:

- for `/detect-text-mask`, use the upstream auto-scan chain with upstream defaults
- for `/inpaint`, the incoming mask is the source of truth
- do not silently re-run detection inside `/inpaint` when `mask_b64` is already provided
- do not silently expand the mask with large dilation unless the upstream code strictly requires it and this is documented in the response warnings

---

## 9. Adapter design

Create a thin adapter layer between FastAPI and upstream modules.

Recommended files:

- `service/app.py`
- `service/schemas.py`
- `service/adapters.py`
- `service/debug_utils.py`

Suggested adapter entrypoints:

```python
def run_manga_cleaner_detect(
    image_bgr: np.ndarray,
    profile: str,
    timeout_sec: int | None = None,
    debug: bool = False,
) -> dict:
    ...


def run_manga_cleaner_inpaint(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    profile: str,
    timeout_sec: int | None = None,
    debug: bool = False,
) -> dict:
    ...
```

Return shape:

```python
{
    "image_bgr": ...,  # only for inpaint
    "mask_u8": ...,    # for detect and optionally for inpaint
    "warnings": [...],
    "metrics": {...},
    "debug_artifacts": {...},
}
```

The adapter should:

- load upstream models once at process startup or lazily via cache
- isolate any path handling
- isolate ONNX Runtime session creation
- convert exceptions into stable service-layer error codes

---

## 10. Upstream code integration guidance

The Windows-side agent should inspect and reuse upstream backend code around:

- model/session loading
- image preprocessing
- OCR logic
- cleaning logic
- path/model discovery helpers

Expected behavior from the reused chain:

- text heatmap / auto-scan generation
- local ROI-based cleaning
- padding to model-friendly shapes
- ONNX Runtime inference
- composition of local cleaned patches back into the full image

If the upstream cleaning code assumes it generated the mask itself, adapt it so that:

- the same downstream cleaning stage can accept:
  - the internally detected mask from `/detect-text-mask`
  - the external mask from macOS
- the external mask from macOS is injected at the stage just before connected components / ROI extraction

This is the key adaptation.

Do **not** discard the upstream per-component / per-tile cleaning stage.

---

## 11. Error handling requirements

The service must distinguish these cases:

- `bad_base64`
- `bad_image`
- `bad_mask`
- `shape_mismatch`
- `backend_unavailable`
- `backend_timeout`
- `model_missing`
- `onnx_runtime_error`
- `internal_error`

Error response format:

```json
{
  "ok": false,
  "error": {
    "code": "model_missing",
    "message": "Required manga-cleaner model file is missing.",
    "detail": {
      "path": "D:\\ai\\manga-cleaner\\models\\lama.onnx"
    }
  }
}
```

Timeout policy:

- keep the request synchronous
- if timeout is exceeded, stop processing if practical and return `backend_timeout`
- if hard cancellation is difficult, at least reject requests that exceed a configured limit before starting

---

## 12. Model and path expectations

The service should not assume the macOS side knows any Windows paths.

All model paths must be local Windows configuration.

Recommended environment variables:

- `MCLEANER_ROOT`
  - root folder of the cloned `manga-cleaner` repo
- `MCLEANER_MODELS_DIR`
  - optional explicit models folder
- `MCLEANER_DEVICE`
  - `cuda` or `cpu`
- `MCLEANER_REQUEST_TIMEOUT_SEC`
  - default request timeout
- `MCLEANER_DEBUG_ARTIFACTS`
  - `0` or `1`

If upstream already has path helpers, prefer reusing them instead of duplicating path logic.

---

## 13. Debug artifacts

When `debug=true`, return lightweight artifacts useful to the macOS side.

Recommended optional debug artifacts:

- `heatmap_b64`
  - upstream text heatmap if available
- `mask_used_b64`
  - the actual binary mask the backend used
- `overlay_b64`
  - image preview with the mask drawn on top

Do not return huge intermediate arrays unless needed.

If saving local files is convenient on Windows, also write:

- `logs/<request_id>/input.png`
- `logs/<request_id>/mask.png`
- `logs/<request_id>/overlay.png`
- `logs/<request_id>/output.png`

This is optional but strongly recommended during rollout.

---

## 14. SAM status and coexistence

Current recommendation:

- do not make SAM part of the new `manga-cleaner` service unless the upstream project itself includes it
- keep SAM only as a separate legacy capability if it is already deployed in the current Windows service

Based on the current upstream public signals we checked:

- the README advertises:
  - auto-scan text
  - AI clean
  - brush / rectangle / lasso tools
- the published `requirements.txt` lists common runtime dependencies such as:
  - `numpy`
  - `onnxruntime_gpu`
  - `opencv_python`
  - `psutil`
  - `pyside6`
  - `pywin32`
- there is no obvious public signal that upstream bundles SAM as a default dependency

So the default assumption for the Windows-side agent should be:

- upstream `manga-cleaner` does **not** ship SAM as a standard part of the core pipeline
- therefore the pure service should **not** add SAM in v1

If later code inspection on Windows proves there is an actual integrated SAM path, then it may be exposed. But until that is proven by the repo code itself, keep the service pure to upstream and leave SAM out.

If the legacy standalone SAM endpoint remains deployed, it solves a different problem:

- SAM:
  - improve mask shape
- manga-cleaner backend:
  - detect text regions and perform localized inpainting

---

## 15. Optional compatibility endpoint

If the Windows-side agent wants a convenience API for manual testing, it may also add:

### `POST /inpaint-from-detect`

Behavior:

1. run `/detect-text-mask`
2. immediately run `/inpaint` with the detected mask
3. return:
   - final image
   - mask
   - optional heatmap

This endpoint is only for convenience.

The macOS agent should still prefer the explicit two-step flow because it is easier to debug and compare.

---

## 16. Test checklist

Before switching macOS to the new backend, verify:

1. `GET /health` works
2. `GET /capabilities` reports `runner=manga_cleaner_onnx`
3. `POST /detect-text-mask` succeeds on a known page
4. `page002` returns a visually reasonable upstream mask
5. result image preserves image dimensions exactly
6. result returns valid PNG base64
7. timeout and missing-model failures return structured JSON
8. debug overlay visually matches the detected or supplied mask
9. a known request with a small synthetic external mask succeeds
10. `page002` succeeds with the current qwen mask

Compare outputs against:

- pure upstream detect + inpaint result
- external-mask inpaint result
- current `inpaint_anything_lama` backend
- same page and same mask where applicable

The main questions are:

- is the upstream auto-scan mask better than our current bbox mask
- is the upstream inpaint result better under the same mask
- does the pure upstream chain already meet the quality bar before any custom tuning

---

## 17. Rollout recommendation

Recommended order:

1. implement `manga_cleaner_onnx` detect + inpaint service on Windows
2. keep the old backend available
3. test the pure upstream two-step flow:
   - `/detect-text-mask`
   - `/inpaint`
4. compare the detected mask against current VLM-derived masks
5. compare the final cleaned result against the old backend
6. if pure upstream is already better, make it the default
7. only then consider any custom mask fusion or tuning

Do **not** stop the current service first.

The migration should be reversible with a config change.
