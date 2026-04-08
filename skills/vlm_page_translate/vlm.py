from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image


DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VLM_MODEL = "qwen3.6-plus"


@dataclass(frozen=True)
class VLMPageTranslateConfig:
    model: str = DEFAULT_VLM_MODEL
    base_url: str = DASHSCOPE_COMPAT_BASE_URL

    # Prefer DASHSCOPE_API_KEY (official docs), fallback to OPENAI_API_KEY (common compat setups).
    api_key_env_primary: str = "DASHSCOPE_API_KEY"
    api_key_env_fallback: str = "OPENAI_API_KEY"

    temperature: float = 0.0
    timeout_seconds: Optional[float] = None

    # Image downscaling (to reduce VLM image tokens).
    # If a limit <= 0, it is disabled.
    #
    # Default policy is aligned with common US comic page scans:
    # - Single page often ~1988x3057
    # - Double-page spreads often ~3976x3057
    #
    # So we cap height to 3057 by default and do NOT cap width.
    # If you want to cap width too (cheaper but may affect double-page spreads), set max_width.
    max_width: int = 0
    max_height: int = 3057

    # Backward-compatible knob: cap the longer side (if enabled).
    # Prefer max_width/max_height for explicit policies.
    max_long_side: int = 0


def _resize_keep_aspect(
    img: Image.Image,
    *,
    max_width: int,
    max_height: int,
    max_long_side: int,
) -> Tuple[Image.Image, int, int]:
    w, h = img.size

    scale = 1.0
    if max_width > 0 and w > max_width:
        scale = min(scale, max_width / float(w))
    if max_height > 0 and h > max_height:
        scale = min(scale, max_height / float(h))
    if max_long_side > 0:
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = min(scale, max_long_side / float(long_side))

    if scale >= 1.0:
        return img, w, h

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Pillow >= 10 uses Image.Resampling
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((new_w, new_h), resample=resample), new_w, new_h


def _prepare_image_data_url(
    image_bytes: bytes,
    *,
    max_width: int,
    max_height: int,
    max_long_side: int,
) -> Tuple[str, int, int, int, int]:
    """Prepare image for VLM.

    Returns:
    - data_url (PNG)
    - orig_w, orig_h
    - sent_w, sent_h (after optional downscale)
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    sent_img, sent_w, sent_h = _resize_keep_aspect(
        img,
        max_width=max_width,
        max_height=max_height,
        max_long_side=max_long_side,
    )

    buf = io.BytesIO()
    sent_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}", orig_w, orig_h, sent_w, sent_h


def _json_loads_best_effort(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _xyxy_to_xywh(box: List[float], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(W), float(x1)))
    x2 = max(0.0, min(float(W), float(x2)))
    y1 = max(0.0, min(float(H), float(y1)))
    y2 = max(0.0, min(float(H), float(y2)))
    x = int(round(min(x1, x2)))
    y = int(round(min(y1, y2)))
    w = int(round(abs(x2 - x1)))
    h = int(round(abs(y2 - y1)))
    return [x, y, max(1, w), max(1, h)]


def _coerce_bbox_xywh(
    bbox: Any,
    W: int,
    H: int,
    *,
    format_hint: str = "xywh",  # xywh|xyxy|auto
) -> Optional[List[int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        nums = [float(x) for x in bbox]
    except Exception:
        return None

    x, y, a, b = nums

    hint = (format_hint or "auto").lower().strip()
    if hint not in {"xywh", "xyxy", "auto"}:
        hint = "auto"

    if hint == "xyxy":
        return _xyxy_to_xywh([x, y, a, b], W, H)

    if hint == "auto":
        # Prefer interpreting as xywh when it fits naturally.
        # If a/b are actually x2/y2, then x+a or y+b often exceeds bounds.
        if (x + a <= W) and (y + b <= H) and (a > 0) and (b > 0):
            hint = "xywh"
        else:
            # If it looks like xyxy inside bounds, use xyxy.
            if (a > x) and (b > y) and (0 <= x <= W) and (0 <= y <= H) and (0 <= a <= W) and (0 <= b <= H):
                hint = "xyxy"
            else:
                hint = "xywh"

    if hint == "xywh":
        x_i = int(round(max(0.0, min(float(W - 1), x))))
        y_i = int(round(max(0.0, min(float(H - 1), y))))
        w_i = int(round(max(1.0, min(float(W), a))))
        h_i = int(round(max(1.0, min(float(H), b))))
        return [x_i, y_i, w_i, h_i]

    return _xyxy_to_xywh([x, y, a, b], W, H)


def _scale_bbox_xywh(bbox_xywh: List[int], *, scale_x: float, scale_y: float, W: int, H: int) -> List[int]:
    x, y, w, h = bbox_xywh
    x2 = int(round(x * scale_x))
    y2 = int(round(y * scale_y))
    w2 = int(round(w * scale_x))
    h2 = int(round(h * scale_y))

    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    w2 = max(1, min(W - x2, w2))
    h2 = max(1, min(H - y2, h2))
    return [x2, y2, w2, h2]


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "boxed", "bubble", "caption", "narration"}:
        return True
    if text in {"false", "0", "no", "n", "embedded", "art", "art_integrated"}:
        return False
    return None


def vlm_extract_translate_page(
    *,
    page_index: int,
    image_bytes: bytes,
    reading_direction: str = "ltr",
    source_language: str = "en",
    target_language: str = "zh",
    retrieval_context: Optional[Dict[str, Any]] = None,
    config: Optional[VLMPageTranslateConfig] = None,
) -> Dict[str, Any]:
    """VLM core step: detect text regions + OCR + translate in one shot.

    Returns envelope:
    {
      ok: bool,
      data: {
        page_index, image_size, regions: [ {region_id, bbox_xywh, src_text, tgt_text, utterance_order?} ]
      },
      metrics: {...},
      error: {...} | null
    }
    """

    cfg = config or VLMPageTranslateConfig()

    api_key = os.getenv(cfg.api_key_env_primary) or os.getenv(cfg.api_key_env_fallback)
    if not api_key:
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {
                "code": "missing_api_key",
                "message": f"Missing API key env: {cfg.api_key_env_primary} (preferred) or {cfg.api_key_env_fallback} (fallback).",
            },
        }

    data_url, orig_w, orig_h, sent_w, sent_h = _prepare_image_data_url(
        image_bytes,
        max_width=cfg.max_width,
        max_height=cfg.max_height,
        max_long_side=cfg.max_long_side,
    )

    system_prompt = (
        "You are a manga/comic translation engine. "
        "Task: detect ALL text regions (speech bubbles, captions, narration boxes), OCR the source text, and translate it. "
        "Return STRICT JSON only (no markdown, no commentary). "
        "Coordinates MUST be pixel coordinates on the image you see in this request. "
        "Prefer bbox format bbox_xywh=[x,y,w,h]. "
        "You may receive retrieval_context with glossary/project/style hints from prior pages or user corrections. "
        "Use those hints to improve translation consistency, especially locked terminology, but NEVER invent regions or OCR text not present in the image. "
        "For each region, also decide has_text_box=true/false. "
        "Use true when the text is inside a visible speech bubble, caption box, narration box, or another clear text container. "
        "Use false when the text is embedded directly into artwork/background, such as titles, SFX, or art-integrated text with no clear container. "
        "Region ids must start from 1 and be stable within this page; order them in reading order. "
        "If no text is present, return an empty regions list."
    )

    user_payload = {
        "task": "vlm_page_translate",
        "page_index": page_index,
        "image_size_original": {"width": orig_w, "height": orig_h},
        "image_size_sent": {"width": sent_w, "height": sent_h},
        "coordinate_space": "sent_image_pixels",
        "reading_direction": reading_direction,
        "source_language": source_language,
        "target_language": target_language,
        "retrieval_context": retrieval_context or {},
        "output_schema": {
            "page_index": "int",
            "image_size": {"width": "int", "height": "int"},
            "regions": [
                {
                    "region_id": "int",
                    "bbox_xywh": "[int,int,int,int]",
                    "has_text_box": "bool",
                    "src_text": "string",
                    "tgt_text": "string",
                    "utterance_order": "int (optional)"
                }
            ],
        },
    }

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    try:
        completion = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                },
            ],
            timeout=cfg.timeout_seconds,
        )
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"orig_width": orig_w, "orig_height": orig_h, "sent_width": sent_w, "sent_height": sent_h},
            "error": {"code": "vlm_request_failed", "message": str(e)},
        }

    content = completion.choices[0].message.content or ""

    try:
        data = _json_loads_best_effort(content)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"orig_width": orig_w, "orig_height": orig_h, "sent_width": sent_w, "sent_height": sent_h},
            "error": {"code": "bad_json", "message": f"VLM did not return JSON: {e}", "detail": content},
        }

    # Normalize/validate minimal fields.
    regions_in = data.get("regions") or data.get("bubbles") or []
    regions_out: List[Dict[str, Any]] = []

    scale_x = float(orig_w) / float(sent_w) if sent_w else 1.0
    scale_y = float(orig_h) / float(sent_h) if sent_h else 1.0

    if isinstance(regions_in, list):
        for idx, r in enumerate(regions_in, start=1):
            if not isinstance(r, dict):
                continue
            region_id = r.get("region_id", idx)
            try:
                region_id = int(region_id)
            except Exception:
                region_id = idx

            if "bbox_xywh" in r:
                bbox_xywh_sent = _coerce_bbox_xywh(r.get("bbox_xywh"), sent_w, sent_h, format_hint="xywh")
            else:
                bbox_xywh_sent = _coerce_bbox_xywh(r.get("bbox"), sent_w, sent_h, format_hint="auto")

            if bbox_xywh_sent is None:
                continue

            bbox_xywh = _scale_bbox_xywh(bbox_xywh_sent, scale_x=scale_x, scale_y=scale_y, W=orig_w, H=orig_h)

            src_text = r.get("src_text") or r.get("text") or ""
            tgt_text = r.get("tgt_text") or r.get("translated_text") or ""
            has_text_box = _coerce_optional_bool(r.get("has_text_box"))
            utterance_order = r.get("utterance_order")
            if utterance_order is not None:
                try:
                    utterance_order = int(utterance_order)
                except Exception:
                    utterance_order = None

            regions_out.append(
                {
                    "region_id": region_id,
                    "bbox_xywh": bbox_xywh,
                    "has_text_box": has_text_box,
                    "src_text": str(src_text),
                    "tgt_text": str(tgt_text),
                    "utterance_order": utterance_order,
                }
            )

    # Ensure deterministic ordering
    regions_out.sort(key=lambda x: (x.get("utterance_order") is None, x.get("utterance_order") or 0, x["region_id"]))

    out_data = {
        "page_index": page_index,
        "image_size": {"width": orig_w, "height": orig_h},
        "image_size_sent": {"width": sent_w, "height": sent_h},
        "sent_to_original_scale": {"x": scale_x, "y": scale_y},
        "regions": regions_out,
        "reading_direction": reading_direction,
        "source_language": source_language,
        "target_language": target_language,
    }

    usage = getattr(completion, "usage", None)
    usage_out = None
    if usage is not None:
        try:
            usage_out = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception:
            usage_out = None

    return {
        "ok": True,
        "data": out_data,
        "metrics": {
            "orig_width": orig_w,
            "orig_height": orig_h,
            "sent_width": sent_w,
            "sent_height": sent_h,
            "downscaled": bool((sent_w, sent_h) != (orig_w, orig_h)),
            "regions": len(regions_out),
            "usage": usage_out,
        },
        "error": None,
    }
