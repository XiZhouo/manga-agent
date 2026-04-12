# Windows Inpainting Service API Spec

This file is the only spec the Windows-side Copilot needs.

Purpose:

- run a FastAPI service on Windows
- accept `image + mask + profile`
- run the chosen inpainting backend
- return the erased page image

Non-goals:

- no VLM
- no OCR
- no translation
- no lettering
- no mask generation from bbox
- no SAM click UI

The macOS Agent already does everything else.

---

## 1. Service role

The Windows service is a synchronous image inpainting service.

Input from macOS:

- original page image
- final page mask
- erase profile

Output back to macOS:

- erased page image
- actual backend used
- profile used

The Windows service must treat the incoming mask as the source of truth.
It must not regenerate, reinterpret, or "improve" the mask in v1.

Important v1 decision:

- use one synchronous `POST /inpaint` request
- do **not** implement:
  - accepted/queued states
  - background jobs
  - polling endpoints
  - result-fetch endpoints

In other words, v1 should be:

- request comes in
- service processes it immediately
- service returns final success/failure JSON in the same response

---

## 2. Required endpoints

### `GET /health`

Purpose:

- tell macOS that the service process is alive

Success response:

```json
{
  "ok": true,
  "name": "InpaintingService",
  "version": "0.1.0",
  "runner": "inpaint_anything_lama"
}
```

Fields:

- `ok`: bool
- `name`: string
- `version`: string
- `runner`: string

Allowed runner examples:

- `inpaint_anything_lama`
- `simple_lama`
- `saicinpainting_cli`
- `unavailable`

---

### `GET /capabilities`

Purpose:

- let macOS know what backend is currently available

Success response:

```json
{
  "ok": true,
  "backend": "lama",
  "runner": "inpaint_anything_lama",
  "profiles": ["conservative", "balanced", "aggressive"],
  "limits": {
    "max_image_side": 0,
    "request_timeout_sec": 180
  }
}
```

Fields:

- `ok`: bool
- `backend`: must be `"lama"`
- `runner`: string
- `profiles`: list of supported logical profiles
- `limits.max_image_side`: integer, `0` means unlimited
- `limits.request_timeout_sec`: integer

---

### `POST /inpaint`

Purpose:

- perform one inpainting request

Request JSON:

```json
{
  "request_id": "job123-page0003-try1",
  "image_b64": "<base64 image bytes>",
  "mask_b64": "<base64 mask png bytes>",
  "backend": "lama",
  "profile": "balanced",
  "timeout_sec": 180
}
```

Fields:

- `request_id`
  - optional but strongly recommended
  - copied back into logs and responses
- `image_b64`
  - required
  - original page image bytes, base64 encoded
- `mask_b64`
  - required
  - binary mask PNG bytes, base64 encoded
  - semantic:
    - `0` = keep
    - `255` = erase
- `backend`
  - required in schema
  - only valid v1 value is `"lama"`
- `profile`
  - required in schema
  - logical values:
    - `conservative`
    - `balanced`
    - `aggressive`
- `timeout_sec`
  - optional
  - requested timeout for this job
  - server may clamp it to a configured min/max

Success response:

```json
{
  "ok": true,
  "request_id": "job123-page0003-try1",
  "image_b64": "<base64 erased png bytes>",
  "backend": "lama:inpaint_anything",
  "profile": "balanced",
  "latency_ms": 8421,
  "warnings": []
}
```

Required success fields:

- `ok`: true
- `request_id`: echo request id if provided, otherwise null or omitted
- `image_b64`: erased image, encoded as PNG bytes then base64
- `backend`: actual backend used
- `profile`: actual logical profile used
- `latency_ms`: total processing time in milliseconds
- `warnings`: list of strings

Expected `backend` values:

- `lama:inpaint_anything`
- `lama:simple_lama`
- `lama:saicinpainting_cli`

Failure response:

```json
{
  "ok": false,
  "request_id": "job123-page0003-try1",
  "error": {
    "code": "backend_timeout",
    "message": "Inpainting backend exceeded the allowed timeout.",
    "detail": {
      "timeout_sec": 180
    }
  },
  "backend": "lama",
  "profile": "balanced",
  "latency_ms": 180034,
  "warnings": []
}
```

Required failure fields:

- `ok`: false
- `request_id`: echo request id if provided
- `error.code`: stable machine-readable code
- `error.message`: human-readable message
- `error.detail`: optional object or string
- `backend`: requested backend or actual backend when known
- `profile`: requested profile
- `latency_ms`: time spent before failing
- `warnings`: list of strings

Important:

- return JSON for both success and failure
- do not rely only on FastAPI default HTML/text errors

---

## 3. Request validation rules

### Image

- decode `image_b64`
- reject invalid base64 with a clear error
- reject invalid image bytes with a clear error

### Mask

- decode `mask_b64`
- reject invalid base64 with a clear error
- reject invalid mask bytes with a clear error
- load as grayscale
- if mask size differs from image size:
  - resize mask to image size using nearest-neighbor
- normalize to binary:
  - values `> 0` become `255`
  - others become `0`

### Backend

- if `backend != "lama"`, reject request

### Profile

- if profile is not in the supported set, reject request

### Timeout

- if `timeout_sec` is missing, use server default
- if too small or too large, clamp to server limits

---

## 4. Timeout and waiting behavior

The request is synchronous.

That means:

- macOS sends one request
- Windows does the full inpainting work
- Windows returns the final result in the same HTTP response

No queue system is needed in v1.

Do not design v1 as:

- `POST /jobs` -> returns accepted
- `GET /jobs/{id}` -> poll status
- `GET /jobs/{id}/result` -> fetch result

That is explicitly out of scope for now.

Recommended timeout behavior:

- service default timeout: `180` seconds
- allow request-level override via `timeout_sec`
- clamp timeout into a safe range, for example:
  - minimum: `30`
  - maximum: `600`

If the backend process exceeds the effective timeout:

- stop the backend subprocess if possible
- return HTTP `504`
- return JSON body with:
  - `ok=false`
  - `error.code="backend_timeout"`

If the service is alive but no backend is available:

- return HTTP `503`
- `error.code="backend_unavailable"`

If the backend crashes during execution:

- return HTTP `500`
- `error.code="backend_execution_failed"`

The macOS caller should be able to distinguish:

- bad request
- temporary service issue
- long-running timeout
- real backend crash

So the service must include these fields in the final response body whenever possible:

- `request_id`
- `latency_ms`
- structured `error.code`

---

## 5. Error codes

The Windows service should use stable error codes like these.

### HTTP 400

- `bad_base64`
- `bad_image`
- `bad_mask`
- `unknown_backend`
- `invalid_profile`
- `image_too_large`

### HTTP 500

- `backend_execution_failed`
- `output_decode_failed`
- `configured_runner_invalid`

### HTTP 503

- `backend_unavailable`

### HTTP 504

- `backend_timeout`

Example failure body:

```json
{
  "ok": false,
  "request_id": "job123-page0003-try1",
  "error": {
    "code": "bad_mask",
    "message": "Failed to decode mask image.",
    "detail": null
  },
  "backend": "lama",
  "profile": "balanced",
  "latency_ms": 14,
  "warnings": []
}
```

---

## 6. Backend abstraction

The Windows code should hide backend-specific details behind one internal function.

Conceptually:

```python
def run_inpaint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    profile: str,
    timeout_sec: int,
) -> tuple[np.ndarray, str]:
    ...
```

Return:

- erased image
- actual backend label

Current preferred implementation target:

- `inpaint_anything_lama`

Allowed actual backend labels:

- `lama:inpaint_anything`
- `lama:simple_lama`
- `lama:saicinpainting_cli`

The service code may internally call:

- a Python API
- a repo script
- a subprocess

That is implementation-specific and not part of the HTTP contract.

---

## 7. Profile mapping

The HTTP API keeps three logical profiles:

- `conservative`
- `balanced`
- `aggressive`

The backend implementation may map them to repo-specific arguments internally.

Example mapping idea:

- `conservative`
  - minimal extra processing
  - safer for dense art
- `balanced`
  - default
- `aggressive`
  - stronger cleanup behavior
  - more suitable for embedded text

Important:

- macOS does not need to know the backend-specific parameter names
- only the logical profile contract matters at the API layer

---

## 8. Logging requirements

For each request, log at least:

- `request_id`
- image width/height
- effective timeout
- profile
- runner/backend used
- success/failure
- latency in ms
- error code if failed

Do not log full base64 payloads.

Example log line:

```text
request_id=job123-page0003-try1 runner=inpaint_anything_lama profile=aggressive size=1988x3057 timeout=180 success=true latency_ms=8421
```

---

## 9. What the macOS side currently expects

The current macOS Agent already does this:

1. build the final mask locally
2. send `image_b64 + mask_b64 + backend + profile`
3. receive the erased image
4. continue lettering locally

The macOS side does **not** need the Windows service to return:

- bbox data
- region data
- mask quality metrics
- overlay images
- translation text

Those remain local to the macOS workflow.

So keep the response small and stable.

---

## 10. Minimal Copilot instruction

You can give Copilot this instruction together with this file:

```text
Implement a standalone FastAPI Windows service that exposes:
- GET /health
- GET /capabilities
- POST /inpaint

POST /inpaint must accept:
- request_id
- image_b64
- mask_b64
- backend="lama"
- profile in {"conservative","balanced","aggressive"}
- timeout_sec

The service must:
- decode image and mask
- resize mask to image size if needed using nearest-neighbor
- binarize the mask
- run a real inpainting backend
- process requests synchronously
- return JSON for both success and failure
- enforce timeout handling
- provide stable error codes

Do not implement queue/accepted/polling endpoints in v1.

Preferred actual backend label is:
- lama:inpaint_anything

Do not implement VLM, OCR, translation, lettering, or mask generation.
```

---

## 11. Final recommendation

For v1, keep the API exactly at the `image + mask -> erased image` level.

That is the cleanest boundary for your current agent and the fastest way to get the full pipeline running.
