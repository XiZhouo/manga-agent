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
DEFAULT_VLM_MODEL = "qwen3-vl-plus"


@dataclass(frozen=True)
class VLMPageTranslateConfig:
    model: str = DEFAULT_VLM_MODEL
    base_url: str = DASHSCOPE_COMPAT_BASE_URL

    # Prefer DASHSCOPE_API_KEY (official docs), fallback to OPENAI_API_KEY (common compat setups).
    api_key_env_primary: str = "DASHSCOPE_API_KEY"
    api_key_env_fallback: str = "OPENAI_API_KEY"

    temperature: float = 0.0
    timeout_seconds: Optional[float] = None
    enable_thinking: Optional[bool] = True
    thinking_budget: Optional[int] = 81920
    vl_high_resolution_images: Optional[bool] = True

    # Image downscaling (to reduce VLM image tokens).
    # If a limit <= 0, it is disabled.
    #
    # Default policy is aligned with common US comic page scans:
    # - Single page often ~1988x3057
    # - Double-page spreads often ~3976x3057
    #
    # Default to full-resolution upload for v1 stability.
    max_width: int = 0
    max_height: int = 0

    # Backward-compatible knob: cap the longer side (if enabled).
    # Prefer max_width/max_height for explicit policies.
    max_long_side: int = 0


def _build_extra_body(cfg: VLMPageTranslateConfig) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if cfg.enable_thinking is not None:
        extra_body["enable_thinking"] = bool(cfg.enable_thinking)
    if cfg.thinking_budget is not None:
        extra_body["thinking_budget"] = int(cfg.thinking_budget)
    if cfg.vl_high_resolution_images is not None:
        extra_body["vl_high_resolution_images"] = bool(cfg.vl_high_resolution_images)
    return extra_body


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


def _xywh_to_xyxy(box: List[float], W: int, H: int) -> List[int]:
    x, y, w, h = box
    x1 = max(0.0, min(float(W - 1), float(x)))
    y1 = max(0.0, min(float(H - 1), float(y)))
    x2 = max(x1 + 1.0, min(float(W), float(x) + max(1.0, float(w))))
    y2 = max(y1 + 1.0, min(float(H), float(y) + max(1.0, float(h))))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _normalize_xyxy(box: List[float], W: int, H: int) -> List[int]:
    x1 = max(0.0, min(float(W - 1), min(float(box[0]), float(box[2]))))
    y1 = max(0.0, min(float(H - 1), min(float(box[1]), float(box[3]))))
    x2 = max(x1 + 1.0, min(float(W), max(float(box[0]), float(box[2]))))
    y2 = max(y1 + 1.0, min(float(H), max(float(box[1]), float(box[3]))))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


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


def _coerce_bbox_xyxy(
    bbox: Any,
    W: int,
    H: int,
    *,
    format_hint: str = "xyxy",  # xywh|xyxy|auto
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

    if hint == "xywh":
        return _xywh_to_xyxy([x, y, a, b], W, H)
    if hint == "xyxy":
        return _normalize_xyxy([x, y, a, b], W, H)

    if (a > x) and (b > y) and (0 <= x <= W) and (0 <= y <= H) and (0 <= a <= W) and (0 <= b <= H):
        return _normalize_xyxy([x, y, a, b], W, H)
    return _xywh_to_xyxy([x, y, a, b], W, H)


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


def _scale_bbox_xyxy(bbox_xyxy: List[int], *, scale_x: float, scale_y: float, W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = bbox_xyxy
    sx1 = int(round(x1 * scale_x))
    sy1 = int(round(y1 * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round(y2 * scale_y))
    sx1 = max(0, min(W - 1, sx1))
    sy1 = max(0, min(H - 1, sy1))
    sx2 = max(sx1 + 1, min(W, sx2))
    sy2 = max(sy1 + 1, min(H, sy2))
    return [sx1, sy1, sx2, sy2]


def _coerce_bbox_xyxy_100(
    bbox: Any,
    W: int,
    H: int,
) -> Optional[List[int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(x) for x in bbox]
    except Exception:
        return None

    x1 = max(0.0, min(100.0, x1))
    y1 = max(0.0, min(100.0, y1))
    x2 = max(0.0, min(100.0, x2))
    y2 = max(0.0, min(100.0, y2))

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)

    px1 = int(round((left / 100.0) * float(W)))
    py1 = int(round((top / 100.0) * float(H)))
    px2 = int(round((right / 100.0) * float(W)))
    py2 = int(round((bottom / 100.0) * float(H)))

    px1 = max(0, min(W - 1, px1))
    py1 = max(0, min(H - 1, py1))
    px2 = max(px1 + 1, min(W, px2))
    py2 = max(py1 + 1, min(H, py2))
    return [px1, py1, px2, py2]


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


def _normalize_page_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"cover", "comic_cover", "front_cover"}:
        return "cover"
    if text in {"info", "information", "credits", "publication_info", "non_content"}:
        return "info"
    return "content"


def _normalize_color(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "black"

    aliases = {
        "blk": "black",
        "white_text": "white",
        "wht": "white",
        "grey": "gray",
        "dark_gray": "gray",
        "light_gray": "gray",
    }
    text = aliases.get(text, text)

    allowed = {
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "purple",
        "orange",
        "gray",
        "brown",
    }
    if text.startswith("#") and len(text) in {4, 7}:
        return text
    if text in allowed:
        return text
    return "black"


def _page_range_from_indices(page_indices: List[int]) -> Dict[str, Any]:
    ordered = sorted(int(x) for x in page_indices)
    return {
        "start_page_index": ordered[0],
        "end_page_index": ordered[-1],
        "page_count": len(ordered),
        "page_indices": ordered,
        "label": f"{ordered[0]}-{ordered[-1]}",
    }


def _normalize_vlm_page_output(
    *,
    page_index: int,
    raw_page: Dict[str, Any],
    orig_w: int,
    orig_h: int,
    sent_w: int,
    sent_h: int,
    reading_direction: str,
    source_language: str,
    target_language: str,
) -> Dict[str, Any]:
    page_type = _normalize_page_type(raw_page.get("page_type"))
    regions_in = raw_page.get("regions") or raw_page.get("bubbles") or []
    regions_out: List[Dict[str, Any]] = []

    scale_sent_to_orig_x = float(orig_w) / float(sent_w) if sent_w else 1.0
    scale_sent_to_orig_y = float(orig_h) / float(sent_h) if sent_h else 1.0
    scale_x = scale_sent_to_orig_x
    scale_y = scale_sent_to_orig_y

    if isinstance(regions_in, list):
        for idx, r in enumerate(regions_in, start=1):
            if not isinstance(r, dict):
                continue
            region_id = r.get("region_id", idx)
            try:
                region_id = int(region_id)
            except Exception:
                region_id = idx

            bbox_xyxy_orig = None
            bbox_xyxy_sent = None

            if bbox_xyxy_orig is None and "bbox_100_xyxy" in r:
                bbox_xyxy_orig = _coerce_bbox_xyxy_100(r.get("bbox_100_xyxy"), orig_w, orig_h)
            if bbox_xyxy_orig is None and "bbox_xyxy_100" in r:
                bbox_xyxy_orig = _coerce_bbox_xyxy_100(r.get("bbox_xyxy_100"), orig_w, orig_h)
            if bbox_xyxy_orig is None and "bbox_norm100_xyxy" in r:
                bbox_xyxy_orig = _coerce_bbox_xyxy_100(r.get("bbox_norm100_xyxy"), orig_w, orig_h)

            if bbox_xyxy_orig is None and "bbox_xyxy_orig" in r:
                bbox_xyxy_orig = _coerce_bbox_xyxy(r.get("bbox_xyxy_orig"), orig_w, orig_h, format_hint="xyxy")
            if bbox_xyxy_orig is None and "bbox_xywh_orig" in r:
                bbox_xyxy_orig = _coerce_bbox_xyxy(r.get("bbox_xywh_orig"), orig_w, orig_h, format_hint="xywh")

            if "bbox_xyxy_sent" in r:
                bbox_xyxy_sent = _coerce_bbox_xyxy(r.get("bbox_xyxy_sent"), sent_w, sent_h, format_hint="xyxy")
            if bbox_xyxy_sent is None and "bbox_xywh_sent" in r:
                bbox_xyxy_sent = _coerce_bbox_xyxy(r.get("bbox_xywh_sent"), sent_w, sent_h, format_hint="xywh")

            if bbox_xyxy_orig is None and "bbox_xyxy" in r:
                bbox_xyxy_sent = _coerce_bbox_xyxy(r.get("bbox_xyxy"), sent_w, sent_h, format_hint="xyxy")

            if bbox_xyxy_orig is None and "bbox_xywh" in r:
                bbox_xyxy_sent = _coerce_bbox_xyxy(r.get("bbox_xywh"), sent_w, sent_h, format_hint="xywh")

            if bbox_xyxy_orig is None and bbox_xyxy_sent is None and "bbox" in r:
                bbox_xyxy_sent = _coerce_bbox_xyxy(r.get("bbox"), sent_w, sent_h, format_hint="auto")

            if bbox_xyxy_orig is None and bbox_xyxy_sent is not None:
                bbox_xyxy_orig = _scale_bbox_xyxy(
                    bbox_xyxy_sent,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    W=orig_w,
                    H=orig_h,
                )

            if bbox_xyxy_orig is None:
                continue

            src_text = r.get("src_text") or r.get("text") or ""
            tgt_text = r.get("tgt_text") or r.get("translated_text") or ""
            has_text_box = _coerce_optional_bool(r.get("has_text_box"))
            color = _normalize_color(r.get("color"))
            utterance_order = r.get("utterance_order")
            if utterance_order is not None:
                try:
                    utterance_order = int(utterance_order)
                except Exception:
                    utterance_order = None

            regions_out.append(
                {
                    "region_id": region_id,
                    "bbox_xyxy": bbox_xyxy_orig,
                    "has_text_box": has_text_box,
                    "color": color,
                    "src_text": str(src_text),
                    "tgt_text": str(tgt_text),
                    "utterance_order": utterance_order,
                }
            )

    if page_type in {"cover", "info"}:
        regions_out = []

    regions_out.sort(key=lambda x: (x.get("utterance_order") is None, x.get("utterance_order") or 0, x["region_id"]))
    return {
        "page_index": page_index,
        "page_type": page_type,
        "original_image_size": {"width": orig_w, "height": orig_h},
        "sent_image_size": {"width": sent_w, "height": sent_h},
        "image_size": {"width": orig_w, "height": orig_h},
        "image_size_sent": {"width": sent_w, "height": sent_h},
        "coordinate_space": "original_image_pixels",
        "sent_to_original_scale": {"x": scale_sent_to_orig_x, "y": scale_sent_to_orig_y},
        "regions": regions_out,
        "reading_direction": reading_direction,
        "source_language": source_language,
        "target_language": target_language,
    }


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
      data: { page_index, original_image_size, sent_image_size, regions: [ {region_id, bbox_xyxy, src_text, tgt_text, utterance_order?} ] },
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
        "你是一个漫画/美漫页面翻译引擎。"
        "你的任务分两步。"
        "第一步，判断这一页属于哪一种页面类型：cover（漫画封面）、info（信息页/版权页/广告页/刊物说明页等非正文页面）、content（漫画正文内容页）。"
        "如果是 cover 或 info，只返回 page_type，并返回空的 regions 列表，不要做 OCR 和翻译。"
        "只有 content 页面才执行文本检测、OCR 和翻译。"
        "对于 content 页面，你必须完成以下流程："
        "把整张图片的宽和高都视为 100 份的尺标，而不是像素坐标；"
        "然后在这个 0 到 100 的统一尺标中检测所有文本区域；"
        "文本框 bbox 必须尽量贴紧文字本身，不要遗漏文字，不要过大，不需要包含整个气泡，不要包含整块大背景，不要包含整格画面，也不要把无关的小图案、装饰块、背景纹理当成文字区域；"
        "最终每个文本区域必须返回 bbox_100_xyxy，语义固定为 [x1,y1,x2,y2]，其中 x1,y1 是左上角，x2,y2 是右下角；"
        "bbox_100_xyxy 的四个数值都以 0 到 100 的尺标表示，可以带小数，不要返回像素坐标；"
        "你还需要为每个文本区域输出 region_id、bbox_100_xyxy、has_text_box、color、src_text、tgt_text，以及可选的 utterance_order。"
        "你可能会收到 retrieval_context，其中包含术语表、项目风格、历史翻译修正等信息。若 retrieval_context 中提供了强约束术语，请优先保证术语一致性，但绝对不要编造图中不存在的文本或区域。"
        "关于 has_text_box：如果文字位于明显的气泡、旁白框、说明框、标题框等可见文本容器中，设为 true；如果文字直接嵌在背景、画面、拟声词效果、标题艺术字中，没有清晰文本框，设为 false。"
        "关于 color：返回文本本身的主要颜色，使用简单标签，例如 black、white、red、blue、yellow、green；如果不确定，默认 black。"
        "region_id 必须从 1 开始，按阅读顺序排序，并且在同一页内保持稳定。"
        "输出要求：只返回 STRICT JSON，不要输出 markdown，不要输出解释，不要输出思考过程，不要输出多余文字。"
    )

    user_payload = {
        "task": "vlm_page_translate",
        "page_index": page_index,
        "original_image_size": {"width": orig_w, "height": orig_h},
        "sent_image_size": {"width": sent_w, "height": sent_h},
        "coordinate_space_required_for_output": "normalized_100_grid",
        "reading_direction": reading_direction,
        "source_language": source_language,
        "target_language": target_language,
        "retrieval_context": retrieval_context or {},
        "bbox_rules": {
            "tight_to_text_ink": True,
            "never_cover_whole_panel": True,
            "for_boxed_text_do_not_cover_whole_bubble": True,
            "large_overlaps_between_regions_are_invalid": True,
        },
        "output_schema": {
            "page_index": "int",
            "page_type": "cover|info|content",
            "original_image_size": {"width": "int", "height": "int"},
            "sent_image_size": {"width": "int", "height": "int"},
            "regions": [
                {
                    "region_id": "int",
                    "bbox_100_xyxy": "[number,number,number,number]",
                    "has_text_box": "bool",
                    "color": "string",
                    "src_text": "string",
                    "tgt_text": "string",
                    "utterance_order": "int (optional)"
                }
            ],
        },
    }
    scaling_instruction = (
        f"这个图像是漫画，我希望你这样做："
        f"请把整张图片的宽和高都视为 100 份尺标，而不是像素。"
        f"然后进行图像识别，获取文本框位置信息。"
        f"文本框要求尽量贴紧文字，不要遗漏文字，但也不要过大，不需要包含整个气泡。"
        f"你返回的每个 bbox 都必须写成 bbox_100_xyxy=[x1,y1,x2,y2]，四个值都使用 0 到 100 的尺标，可以带小数。"
        f"不要返回像素坐标，也不要自己换算成 {orig_w}x{orig_h}。"
        f"我会在本地把你返回的 100 份尺标坐标再换算回原图像素尺寸 {orig_w}x{orig_h}。"
        f"最终返回一个 JSON，记录每一个文本框 bbox_100_xyxy、框内原文字以及翻译中文后的文本。"
        f"如果页面是封面或信息页，则仍按标准 JSON 格式返回，但 regions 为空。"
        f"再次强调：只返回 100 份尺标坐标，不要返回像素坐标。"
    )

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)
    extra_body = _build_extra_body(cfg)

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
                        {"type": "text", "text": scaling_instruction},
                        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                },
            ],
            timeout=cfg.timeout_seconds,
            extra_body=extra_body,
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

    out_data = _normalize_vlm_page_output(
        page_index=page_index,
        raw_page=data,
        orig_w=orig_w,
        orig_h=orig_h,
        sent_w=sent_w,
        sent_h=sent_h,
        reading_direction=reading_direction,
        source_language=source_language,
        target_language=target_language,
    )

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
            "regions": len(out_data["regions"]),
            "usage": usage_out,
        },
        "error": None,
    }


def vlm_extract_translate_pages(
    *,
    pages: List[Dict[str, Any]],
    reading_direction: str = "ltr",
    source_language: str = "en",
    target_language: str = "zh",
    config: Optional[VLMPageTranslateConfig] = None,
) -> Dict[str, Any]:
    """Batch VLM core step for a small page window (recommended <= 10 pages)."""

    cfg = config or VLMPageTranslateConfig()

    if not pages:
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {"code": "missing_pages", "message": "pages is empty"},
        }

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

    prepared_pages: List[Dict[str, Any]] = []
    content_blocks: List[Dict[str, Any]] = []
    page_indices: List[int] = []

    for order, item in enumerate(pages):
        if not isinstance(item, dict):
            continue
        try:
            page_index = int(item.get("page_index"))
        except Exception:
            continue
        image_bytes = item.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            return {
                "ok": False,
                "data": None,
                "metrics": {},
                "error": {"code": "missing_image_bytes", "message": f"Page {page_index} is missing image bytes."},
            }
        data_url, orig_w, orig_h, sent_w, sent_h = _prepare_image_data_url(
            bytes(image_bytes),
            max_width=cfg.max_width,
            max_height=cfg.max_height,
            max_long_side=cfg.max_long_side,
        )
        retrieval_context = item.get("retrieval_context")
        prepared_pages.append(
            {
                "slot_index": order,
                "page_index": page_index,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "sent_width": sent_w,
                "sent_height": sent_h,
                "retrieval_context": retrieval_context if isinstance(retrieval_context, dict) else {},
            }
        )
        page_indices.append(page_index)
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    if not prepared_pages:
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {"code": "no_valid_pages", "message": "No valid pages were provided to VLM batch."},
        }

    page_range = _page_range_from_indices(page_indices)
    system_prompt = (
        "你是一个漫画/美漫多页翻译引擎。"
        "你将一次收到多张页面图像，每张图像都对应 pages_meta 中的一页。"
        "对于每一页，你都必须先判断页面类型：cover（漫画封面）、info（信息页/版权页/广告页/刊物说明页等非正文页面）、content（漫画正文内容页）。"
        "如果某一页是 cover 或 info，只返回该页的 page_type，并返回空的 regions 列表，不要做 OCR 和翻译。"
        "只有 content 页面才执行文本检测、OCR 和翻译。"
        "对于每一页 content 页面，你都必须完成以下流程："
        "把每一张图片的宽和高都视为 100 份的尺标，而不是像素坐标；"
        "然后在这个 0 到 100 的统一尺标中检测所有文本区域，得到 bbox_100_xyxy=[x1,y1,x2,y2]；"
        "文本框 bbox 必须尽量贴紧文字本身，不要遗漏文字，不要过大，不需要包含整个气泡；"
        "bbox_100_xyxy 的四个数值都以 0 到 100 的尺标表示，可以带小数，不要返回像素坐标。"
        "你还需要为每个文本区域输出 region_id、bbox_100_xyxy、has_text_box、color、src_text、tgt_text，以及可选的 utterance_order。"
        "你可能会收到 retrieval_context，其中包含术语表、项目风格、历史翻译修正等信息。若 retrieval_context 中提供了强约束术语，请优先保证术语一致性，但绝对不要编造图中不存在的文本或区域。"
        "关于 has_text_box：如果文字位于明显的气泡、旁白框、说明框、标题框等可见文本容器中，设为 true；如果文字直接嵌在背景、画面、拟声词效果、标题艺术字中，没有清晰文本框，设为 false。"
        "关于 color：返回文本本身的主要颜色，使用简单标签，例如 black、white、red、blue、yellow、green；如果不确定，默认 black。"
        "region_id 必须从 1 开始，在每一页内部按阅读顺序排序，并且同一页内保持稳定。"
        "输出要求：只返回 STRICT JSON，不要输出 markdown，不要输出解释，不要输出思考过程，不要输出多余文字。"
    )

    user_payload = {
        "task": "vlm_page_translate_batch",
        "page_range": page_range,
        "reading_direction": reading_direction,
        "source_language": source_language,
        "target_language": target_language,
        "coordinate_space_required_for_output": "normalized_100_grid",
        "bbox_rules": {
            "tight_to_text_ink": True,
            "never_cover_whole_panel": True,
            "for_boxed_text_do_not_cover_whole_bubble": True,
            "large_overlaps_between_regions_are_invalid": True,
        },
        "pages_meta": [
            {
                "slot_index": item["slot_index"],
                "page_index": item["page_index"],
                "original_image_size": {"width": item["orig_width"], "height": item["orig_height"]},
                "sent_image_size": {"width": item["sent_width"], "height": item["sent_height"]},
                "retrieval_context": item["retrieval_context"],
            }
            for item in prepared_pages
        ],
        "output_schema": {
            "page_range": {
                "start_page_index": "int",
                "end_page_index": "int",
                "page_count": "int",
                "page_indices": "int[]",
                "label": "string",
            },
            "pages": [
                {
                    "page_index": "int",
                    "page_type": "cover|info|content",
                    "regions": [
                        {
                            "region_id": "int",
                            "bbox_100_xyxy": "[number,number,number,number]",
                            "has_text_box": "bool",
                            "color": "string",
                            "src_text": "string",
                            "tgt_text": "string",
                            "utterance_order": "int (optional)",
                        }
                    ],
                }
            ],
        },
    }
    page_size_lines = "\n".join(
        f"- 第 {item['page_index']} 页最终目标尺寸：{item['orig_width']}x{item['orig_height']}"
        for item in prepared_pages
    )
    batch_scaling_instruction = (
        "这是一个多页漫画翻译任务。\n"
        "对于每一页，请把整张图片的宽和高都视为 100 份尺标，而不是像素。\n"
        "然后进行图像识别，获取文本框位置信息，格式为 bbox_100_xyxy=[x1,y1,x2,y2]。\n"
        "文本框要求尽量贴紧文字，不要遗漏文字，但也不要过大，不需要包含整个气泡。\n"
        "每个 bbox_100_xyxy 的四个值都使用 0 到 100 的尺标，可以带小数，不要返回像素坐标。\n"
        "这些页面在本地会被换算回原图像素尺寸，你不用自己做像素换算。\n"
        f"对应原图尺寸如下，仅供你理解页面比例，不要直接返回像素坐标：\n{page_size_lines}\n"
        "最终返回的每一页都只返回 bbox_100_xyxy，不要返回像素 bbox_xyxy。"
    )

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)
    extra_body = _build_extra_body(cfg)

    try:
        completion = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": batch_scaling_instruction},
                        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    ]
                    + content_blocks,
                },
            ],
            timeout=cfg.timeout_seconds,
            extra_body=extra_body,
        )
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"page_range": page_range},
            "error": {"code": "vlm_request_failed", "message": str(e)},
        }

    content = completion.choices[0].message.content or ""
    try:
        data = _json_loads_best_effort(content)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"page_range": page_range},
            "error": {"code": "bad_json", "message": f"VLM did not return JSON: {e}", "detail": content},
        }

    raw_pages = data.get("pages") or []
    if not isinstance(raw_pages, list):
        raw_pages = []
    raw_by_index = {}
    for item in raw_pages:
        if not isinstance(item, dict):
            continue
        try:
            raw_by_index[int(item.get("page_index"))] = item
        except Exception:
            continue

    pages_out: List[Dict[str, Any]] = []
    per_page_metrics: List[Dict[str, Any]] = []
    total_regions = 0
    for meta in prepared_pages:
        page_index = meta["page_index"]
        normalized = _normalize_vlm_page_output(
            page_index=page_index,
            raw_page=raw_by_index.get(page_index) or {"regions": []},
            orig_w=meta["orig_width"],
            orig_h=meta["orig_height"],
            sent_w=meta["sent_width"],
            sent_h=meta["sent_height"],
            reading_direction=reading_direction,
            source_language=source_language,
            target_language=target_language,
        )
        pages_out.append(normalized)
        total_regions += len(normalized["regions"])
        per_page_metrics.append(
            {
                "page_index": page_index,
                "orig_width": meta["orig_width"],
                "orig_height": meta["orig_height"],
                "sent_width": meta["sent_width"],
                "sent_height": meta["sent_height"],
                "downscaled": bool((meta["sent_width"], meta["sent_height"]) != (meta["orig_width"], meta["orig_height"])),
                "regions": len(normalized["regions"]),
            }
        )

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
        "data": {
            "page_range": page_range,
            "pages": pages_out,
            "reading_direction": reading_direction,
            "source_language": source_language,
            "target_language": target_language,
        },
        "metrics": {
            "page_range": page_range,
            "page_count": len(pages_out),
            "total_regions": total_regions,
            "pages": per_page_metrics,
            "usage": usage_out,
        },
        "error": None,
    }
