# Skill1: Panel & Text Region Detection (Manga-Bubble-YOLO ONNX)

## Purpose

Detect all text regions on a single manga/comic page image using Manga-Bubble-YOLO (ONNX) and return structured bounding boxes for downstream OCR / translation / lettering.

Baseline behavior (current):

* Treat the whole page as one panel.
* Detect text regions on the page.
* Assign each region a `region_id` (1..N) within the page.
* Attach `page_index` to every region for traceability.
* Return `region_type="text"` for all regions (future: bubble/caption/sound_effect).

---

## Inputs

### Required

* **page_index** (int)

  * Page number from Skill0 (0-based).
* **image_bytes** (bytes)

  * Normalized page image bytes (RGB recommended).
* **model_choice** (string)

  * Model alias (`yolo26n` / `yolo26s`) OR a direct filesystem path to an ONNX model.

### Optional

* **source_filename** (string | null)

  * Filename from Skill0 for traceability (e.g., `page_0007.png`).
* **conf_threshold** (float)

  * Confidence threshold (default: 0.25).
* **iou_threshold** (float)

  * IoU threshold for NMS (default: 0.45).
* **device** (string | null)

  * `"cpu"` or `"0"` (GPU id).

---

## Output Schema

Top-level **PageDetectionResult**:

* **page_index** (int)
* **source_filename** (string | null)
* **image_size** (object)

  * **width** (int)
  * **height** (int)
* **panels** (array of **PanelData**)

  * **bbox** (array[int]): `[x, y, w, h]` (original image pixels)
  * **text_regions** (array of **TextRegion**)

    * **region_id** (int): 1..N (per page)
    * **page_index** (int): equals input `page_index`
    * **bbox** (array[int]): `[x, y, w, h]` (original image pixels)
    * **score** (float): confidence score
    * **region_type** (string): `"text"` (baseline)

Notes:

* Baseline returns exactly one panel whose bbox is the full page: `[0, 0, width, height]`.

---

## Input Example (JSON)

**Example input object** (conceptual; `image_bytes` is bytes, not JSON):

```text
{
  page_index: 7,
  source_filename: "page_0007.png",
  model_choice: "yolo26s",
  conf_threshold: 0.25,
  iou_threshold: 0.45,
  device: "cpu",
  image_bytes: <BYTES>
}
```

---

## Output Example (JSON)

```json
{
  "page_index": 7,
  "source_filename": "page_0007.png",
  "image_size": { "width": 1600, "height": 2400 },
  "panels": [
    {
      "bbox": [0, 0, 1600, 2400],
      "text_regions": [
        {
          "region_id": 1,
          "page_index": 7,
          "bbox": [120, 210, 560, 310],
          "score": 0.92,
          "region_type": "text"
        },
        {
          "region_id": 2,
          "page_index": 7,
          "bbox": [300, 1350, 420, 260],
          "score": 0.81,
          "region_type": "text"
        }
      ]
    }
  ]
}
```

---

## Model Catalog (ONNX Only)

This project only supports ONNX models (no PyTorch `.pt`).

### Model: yolo26n.onnx (alias: yolo26n)

* **Type**: YOLO26 Nano (ONNX)
* **Strengths**:

  * Fastest inference
  * Lowest memory footprint
  * Good for CPU / large batch processing
* **Weaknesses**:

  * May miss tiny text
  * Lower recall on stylized SFX / low-contrast regions
* **Path** (fill in your repo path):

  * `/Users/jay/Code/agent1/models/Manga-Bubble-YOLO/onnx/yolo26n.onnx`

### Model: yolo26s.onnx (alias: yolo26s) — Recommended Default

* **Type**: YOLO26 Small (ONNX)
* **Strengths**:

  * Better recall/precision trade-off for typical manga pages
  * More robust on small fonts than Nano
* **Weaknesses**:

  * Slower than Nano
  * Higher memory usage
* **Path** (fill in your repo path):

  * `/Users/jay/Code/agent1/models/Manga-Bubble-YOLO/onnx/yolo26s.onnx`

---

## Model Selection Policy (Agent / Pipeline Controller)

Suggested defaults:

* If speed is the priority (CPU, large batches): choose `yolo26n`
* If quality is the priority (best recall on typical pages): choose `yolo26s`

The controller can either:

1. pick a fixed default (`yolo26s`), or
2. call the MCP tool `list_available_models` to fetch the catalog and decide at runtime.

---

## MCP Tools

### list_available_models()

Returns a machine-readable catalog of model aliases, paths, and descriptions.

Example output:

```json
{
  "default": "yolo26s",
  "models": [
    {
      "alias": "yolo26n",
      "path": "PATH/TO/yolo26n.onnx",
      "desc": "YOLO26 Nano (ONNX): fastest; may miss tiny/stylized text."
    },
    {
      "alias": "yolo26s",
      "path": "PATH/TO/yolo26s.onnx",
      "desc": "YOLO26 Small (ONNX): recommended default; better recall/precision trade-off."
    }
  ]
}
```

### detect(page_index, image_bytes, model_choice, source_filename, conf_threshold, iou_threshold, device)

Runs detection for a single page image and returns **PageDetectionResult** (see Output Schema).

---

## Notes / Design Rationale

* `page_index` should come from Skill0 output and must be passed in by the pipeline controller.
* The skill does not infer `page_index` from filename because filenames may be absent or inconsistent.
* We rely on Ultralytics for preprocessing, NMS, and mapping bounding boxes back to original image coordinates.

---

## Failure Modes

* If no regions are detected, return a valid result with `text_regions: []`.
* If the model cannot be loaded (missing file / incompatible ONNX), raise an error with a clear message including `model_choice`.
