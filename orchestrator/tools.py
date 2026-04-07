from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from .state import PageState


class Tools:
    """
    这里是对 skills 的调用封装层。

    当前先用 mock 实现，便于把 orchestrator 主干跑通。
    以后你可以把这些方法替换成：
    - MCP client 调用
    - LangChain Tool 调用
    - HTTP API 调用
    """

    def run_detection(self, page: PageState) -> Dict[str, Any]:
        """
        Skill1: 文本区域检测。

        v0:
        - 如果 `page.image_ref` 指向存在的图片文件，则调用真实的 Manga-Bubble-YOLO ONNX 检测。
        - 否则回退到 mock（用于 orchestrator demo / 无输入文件场景）。

        返回值模拟结构：
        {
          "ok": True,
          "regions": [
              {"region_id": 1, "bbox": [...], "region_type": "text"},
              ...
          ],
          "bbox_error": False,
        }
        """
        if page.image_ref:
            path = Path(page.image_ref)
            if path.exists() and path.is_file():
                try:
                    from skills.panel_text_detection.detection import detect_text_regions

                    model_choice = os.getenv("MANGA_AGENT_DETECTION_MODEL", "yolo26s")
                    imgsz = int(os.getenv("MANGA_AGENT_DETECTION_IMGSZ", "1280"))
                    conf = float(os.getenv("MANGA_AGENT_DETECTION_CONF", "0.25"))
                    iou = float(os.getenv("MANGA_AGENT_DETECTION_IOU", "0.45"))
                    device = os.getenv("MANGA_AGENT_DETECTION_DEVICE") or "cpu"

                    image_bytes = path.read_bytes()

                    def _call(model: str, conf_threshold: float) -> Dict[str, Any]:
                        return detect_text_regions(
                            page_index=page.page_index,
                            image_bytes=image_bytes,
                            model_choice=model,
                            source_filename=page.source_filename,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou,
                            imgsz=imgsz,
                            device=device,
                        )

                    # Fallback policy (only if 0 regions):
                    # 1) configured model + configured conf
                    # 2) configured model + lower conf (0.10)
                    # 3) fallback model yolo26n + lower conf (0.10)
                    raw = _call(model_choice, conf)

                    def _extract(raw_out: Dict[str, Any]) -> list[dict[str, Any]]:
                        out = []
                        for panel in raw_out.get("panels", []) or []:
                            for r in panel.get("text_regions", []) or []:
                                out.append(
                                    {
                                        "region_id": int(r["region_id"]),
                                        "bbox": r["bbox"],
                                        "region_type": r.get("region_type", "text"),
                                        "score": r.get("score"),
                                    }
                                )
                        return out

                    text_regions = _extract(raw)
                    if not text_regions:
                        raw = _call(model_choice, conf_threshold=0.10)
                        text_regions = _extract(raw)
                    if not text_regions and model_choice != "yolo26n":
                        raw = _call("yolo26n", conf_threshold=0.10)
                        text_regions = _extract(raw)

                    return {
                        "ok": True,
                        "regions": [
                            {"region_id": r["region_id"], "bbox": r["bbox"], "region_type": r["region_type"]}
                            for r in text_regions
                        ],
                        "bbox_error": False,
                        "raw": raw,
                    }
                except Exception as e:
                    return {"ok": False, "error": {"code": "detection_exception", "message": str(e)}}

        # fallback mock
        return {
            "ok": True,
            "regions": [
                {"region_id": 1, "bbox": [100, 100, 300, 120], "region_type": "text"},
                {"region_id": 2, "bbox": [120, 300, 280, 140], "region_type": "text"},
            ],
            "bbox_error": False,
        }

    def run_vlm_page(
        self,
        page: PageState,
        *,
        reading_direction: str = "ltr",
        source_language: str = "en",
        target_language: str = "zh",
        force_fullres: bool = False,
    ) -> Dict[str, Any]:
        """VLM core: detect + OCR + translate on a page in one call.

        This is the new "VLM workflow" path:
        - InputNormalization -> VLM core -> Lettering

        Returned structure:
        {
          ok: bool,
          regions: [ {region_id, bbox, region_type, src_text, tgt_text} ],
          raw: <vendor response>
        }
        """

        if not page.image_ref:
            return {"ok": False, "error": {"code": "missing_image_ref", "message": "page.image_ref is empty"}}

        path = Path(page.image_ref)
        if not (path.exists() and path.is_file()):
            return {"ok": False, "error": {"code": "missing_image_file", "message": f"Image not found: {path}"}}

        try:
            from skills.vlm_page_translate.vlm import (
                VLMPageTranslateConfig,
                vlm_extract_translate_page,
            )

            model = os.getenv("MANGA_AGENT_VLM_MODEL", "qwen3.6-plus")
            base_url = os.getenv("MANGA_AGENT_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            temperature = float(os.getenv("MANGA_AGENT_VLM_TEMPERATURE", "0.0"))
            # Resize policy defaults (aligned with common US-comic page scans):
            # - cap height to 3057
            # - do not cap width (keeps double-page spreads)
            max_width = int(os.getenv("MANGA_AGENT_VLM_MAX_WIDTH", "0"))
            max_height = int(os.getenv("MANGA_AGENT_VLM_MAX_HEIGHT", "3057"))
            # Optional extra safety cap
            max_long_side = int(os.getenv("MANGA_AGENT_VLM_MAX_LONG_SIDE", "0"))

            if force_fullres:
                cfg = VLMPageTranslateConfig(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_width=0,
                    max_height=0,
                    max_long_side=0,
                )
            else:
                cfg = VLMPageTranslateConfig(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_width=max_width,
                    max_height=max_height,
                    max_long_side=max_long_side,
                )

            raw = vlm_extract_translate_page(
                page_index=page.page_index,
                image_bytes=path.read_bytes(),
                reading_direction=reading_direction,
                source_language=source_language,
                target_language=target_language,
                config=cfg,
            )

            # Fallback policy (quality-first):
            # If we downscaled and got 0 regions, retry once at full resolution.
            # This costs more tokens but avoids missing-page failures.
            regions_n = int((raw.get("metrics") or {}).get("regions") or 0)
            downscaled = bool((raw.get("metrics") or {}).get("downscaled") is True)
            if (not force_fullres) and raw.get("ok") and downscaled and regions_n == 0:
                cfg_full = VLMPageTranslateConfig(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_width=0,
                    max_height=0,
                    max_long_side=0,
                )
                raw2 = vlm_extract_translate_page(
                    page_index=page.page_index,
                    image_bytes=path.read_bytes(),
                    reading_direction=reading_direction,
                    source_language=source_language,
                    target_language=target_language,
                    config=cfg_full,
                )
                if raw2.get("ok"):
                    raw = raw2

            if not raw.get("ok"):
                return {"ok": False, "error": raw.get("error"), "raw": raw}

            data = raw.get("data") or {}
            regions_out = []
            for r in data.get("regions", []) or []:
                regions_out.append(
                    {
                        "region_id": int(r.get("region_id")),
                        "bbox": r.get("bbox_xywh"),
                        "region_type": "text",
                        "src_text": r.get("src_text", ""),
                        "tgt_text": r.get("tgt_text", ""),
                    }
                )

            metrics = raw.get("metrics") or {}
            return {
                "ok": True,
                "regions": regions_out,
                "bbox_error": False,
                "raw": raw,
                "vlm_metrics": {
                    "orig": (metrics.get("orig_width"), metrics.get("orig_height")),
                    "sent": (metrics.get("sent_width"), metrics.get("sent_height")),
                    "downscaled": metrics.get("downscaled"),
                    "usage": metrics.get("usage"),
                },
            }
        except Exception as e:
            return {"ok": False, "error": {"code": "vlm_exception", "message": str(e)}}

    def run_vlm_judge_text_only(
        self,
        *,
        page_index: int,
        target_language: str,
        reading_direction: str,
        regions: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Text-only quality judge for VLM outputs (no image upload).

        Returns envelope:
        {
          ok: bool,
          data: { decision, reason_code, explanation, confidence, warnings[] },
          metrics: { usage? },
          error: {code,message,detail?} | null
        }
        """

        try:
            from .vlm_judge import judge_vlm_page_text_only

            return judge_vlm_page_text_only(
                page_index=page_index,
                target_language=target_language,
                reading_direction=reading_direction,
                regions=regions,
                metrics=metrics,
                warnings=warnings,
            )
        except Exception as e:
            return {"ok": False, "data": None, "metrics": {}, "error": {"code": "judge_exception", "message": str(e)}}

    def run_vlm_translation_revision(
        self,
        *,
        pages: List[Dict[str, Any]],
        user_request: str,
        reference_page_number: Optional[int],
        scope: str,
        from_side: str,
        from_text: Optional[str],
        to_text: Optional[str],
        reading_direction: str,
        target_language: str,
        glossary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Text-only revision for VLM translation outputs (no image upload, no VLM rerun).

        Input/output schema matches VLM-style pages->regions with src/tgt.
        """

        try:
            from skills.vlm_translation_revision.revise import (
                VLMTranslationRevisionConfig,
                revise_vlm_translation,
            )

            model = os.getenv("MANGA_AGENT_REVISION_MODEL", "qwen3-max")
            temperature = float(os.getenv("MANGA_AGENT_REVISION_TEMPERATURE", "0.0"))
            max_candidate_pages = int(os.getenv("MANGA_AGENT_REVISION_MAX_CANDIDATE_PAGES", "12"))
            max_patches = int(os.getenv("MANGA_AGENT_REVISION_MAX_PATCHES", "64"))

            cfg = VLMTranslationRevisionConfig(
                model=model,
                temperature=temperature,
                max_candidate_pages=max_candidate_pages,
                max_patches=max_patches,
            )

            return revise_vlm_translation(
                pages=pages,
                user_request=user_request,
                reference_page_number=reference_page_number,
                scope=scope,
                from_side=from_side,
                from_text=from_text,
                to_text=to_text,
                reading_direction=reading_direction,
                target_language=target_language,
                glossary=glossary,
                config=cfg,
            )
        except Exception as e:
            return {"ok": False, "data": None, "metrics": {}, "error": {"code": "revision_exception", "message": str(e)}}

    def run_ocr(self, page: PageState) -> Dict[str, Any]:
        """
        模拟 Skill2: OCR。

        故意让某些区域第一次 OCR 置信度偏低，便于演示自动重试。
        """
        results = {}
        for region_id, region in page.regions.items():
            if region_id == 2 and region.retry_count_ocr == 0:
                results[region_id] = {
                    "text": "unclear text",
                    "confidence": 0.55,
                }
            else:
                results[region_id] = {
                    "text": f"sample text page{page.page_index} region{region_id}",
                    "confidence": 0.93,
                }

        return {"ok": True, "results": results}

    def run_translation(self, page: PageState, target_language: str) -> Dict[str, Any]:
        """
        Skill4: 翻译。

        v0:
        - 如果目标语言是中文（zh），优先走 LangChain + DashScope(Qwen) 的真实翻译能力。
        - 其它情况先走 mock（后续再补齐多语言）。
        """
        if target_language == "zh":
            try:
                from skills.translation_en2zh.translate import translate_utterances_en2zh

                utterances = [
                    {
                        "region_id": region_id,
                        "clean_text": (region.ocr_text or "").strip(),
                        "speaker_id": None,
                        "utterance_type": "unknown",
                    }
                    for region_id, region in sorted(page.regions.items())
                ]

                out = translate_utterances_en2zh(
                    utterances=utterances,
                    page_context="",
                    glossary=[],
                )
                if not out.get("ok"):
                    return {"ok": False, "error": out.get("error")}

                results: Dict[int, Dict[str, Any]] = {}
                for region_id, item in out.get("results", {}).items():
                    results[int(region_id)] = {
                        "translated_text": item.get("translated_text", ""),
                        "confidence": 0.90,
                    }

                return {"ok": True, "results": results, "consistent": True, "warnings": out.get("warnings", [])}
            except Exception as e:
                return {"ok": False, "error": {"code": "translation_exception", "message": str(e)}}

        # fallback mock
        results = {}
        for region_id, region in page.regions.items():
            src = region.ocr_text or ""
            results[region_id] = {
                "translated_text": f"[{target_language}] {src}",
                "confidence": 0.90,
            }

        return {"ok": True, "results": results, "consistent": True}

    def run_lettering(self, page: PageState) -> Dict[str, Any]:
        """
        模拟 Skill5: 自动嵌字。

        这里只简单用“翻译文本过长就可能溢出”来模拟失败。
        """
        results = {}
        for region_id, region in page.regions.items():
            translated = region.translated_text or ""
            overflow = len(translated) > 60
            results[region_id] = {
                "render_ok": not overflow,
                "overflow": overflow,
            }

        return {"ok": True, "results": results}

    def ask_user(self, reason: str) -> Dict[str, Any]:
        """
        真实系统里这里可以改成：
        - 推送消息
        - 写入通知表
        - 返回 API response
        """
        return {"ok": True, "message": reason}