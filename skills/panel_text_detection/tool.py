from langchain_core.tools import tool
from skills.panel_text_detection.detection import detect_text_regions

DEFAULT_MODEL_CHOICE = "yolo26s"  # alias in MODEL_CATALOG or a real filesystem path

@tool
def panel_text_detection(
    page_index: int,
    image_bytes: bytes,
    model_choice: str = DEFAULT_MODEL_CHOICE,
    source_filename: str | None = None,
) -> dict:
    return detect_text_regions(
        page_index=page_index,
        image_bytes=image_bytes,
        model_choice=model_choice,
        source_filename=source_filename,
    )
