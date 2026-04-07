from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


def _repo_root() -> Path:
    # skills/panel_text_detection/detection.py -> repo root
    return Path(__file__).resolve().parents[2]


MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "yolo26n": {
        "path": str(_repo_root() / "models" / "Manga-Bubble-YOLO" / "onnx" / "yolo26n.onnx"),
        "desc": "YOLO26-Nano (ONNX): fastest, lowest memory; may miss tiny/stylized text.",
    },
    "yolo26s": {
        "path": str(_repo_root() / "models" / "Manga-Bubble-YOLO" / "onnx" / "yolo26s.onnx"),
        "desc": "YOLO26-Small (ONNX): recommended default; better recall/precision trade-off.",
    },
}

DEFAULT_MODEL_ALIAS = "yolo26s"


@dataclass
class TextRegion:
    region_id: int
    page_index: int
    bbox: List[int]   # [x, y, w, h] in original image pixels
    score: float
    region_type: str = "text"


@dataclass
class PanelData:
    bbox: List[int]   # [x, y, w, h]
    text_regions: List[TextRegion]


@dataclass
class PageDetectionResult:
    page_index: int
    source_filename: Optional[str]
    image_size: Dict[str, int]  # {"width": W, "height": H}
    panels: List[PanelData]


def list_models() -> Dict[str, Any]:
    return {
        "default": DEFAULT_MODEL_ALIAS,
        "models": [
            {"alias": k, "path": v["path"], "desc": v["desc"]}
            for k, v in MODEL_CATALOG.items()
        ],
    }


def _load_image(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    return np.array(img), w, h


def _resolve_model_path(model_choice: str) -> str:
    # allow alias or direct path
    if model_choice in MODEL_CATALOG:
        return MODEL_CATALOG[model_choice]["path"]
    return model_choice


def _xyxy_to_xywh(xyxy: List[float], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = xyxy
    # clamp to image
    x1 = max(0.0, min(float(W), x1))
    x2 = max(0.0, min(float(W), x2))
    y1 = max(0.0, min(float(H), y1))
    y2 = max(0.0, min(float(H), y2))

    x = int(round(min(x1, x2)))
    y = int(round(min(y1, y2)))
    w = int(round(abs(x2 - x1)))
    h = int(round(abs(y2 - y1)))
    return [x, y, max(1, w), max(1, h)]


def detect_text_regions(
    page_index: int,
    image_bytes: bytes,
    model_choice: str = DEFAULT_MODEL_ALIAS,
    *,
    source_filename: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 1280,
    device: Optional[str] = None,  # "cpu" or "0"
) -> Dict[str, Any]:
    """
    Baseline:
      - Treat entire page as one panel
      - Return all detected text regions as region_type="text"
      - region_id starts at 1 for each page
    """
    image_np, W, H = _load_image(image_bytes)
    model_path = _resolve_model_path(model_choice)

    model = YOLO(model_path)
    results = model.predict(
        source=image_np,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )

    regions: List[TextRegion] = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        r0 = results[0]
        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()

        for idx, (box, score) in enumerate(zip(xyxy, conf), start=1):
            bbox_xywh = _xyxy_to_xywh([float(x) for x in box], W, H)
            regions.append(
                TextRegion(
                    region_id=idx,
                    page_index=page_index,
                    bbox=bbox_xywh,
                    score=float(score),
                    region_type="text",
                )
            )

    page_panel = PanelData(bbox=[0, 0, W, H], text_regions=regions)
    out = PageDetectionResult(
        page_index=page_index,
        source_filename=source_filename,
        image_size={"width": W, "height": H},
        panels=[page_panel],
    )
    return asdict(out)