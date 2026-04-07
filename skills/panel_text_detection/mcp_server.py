from fastmcp import FastMCP
from skills.panel_text_detection.detection import detect_text_regions, list_models

mcp = FastMCP("PanelTextDetection")

@mcp.tool()
def list_available_models():
    """MCP 接口：返回可用的 Manga-Bubble-YOLO ONNX 模型目录（含别名与路径占位）"""
    return list_models()

@mcp.tool()
def detect(
    page_index: int,
    image_bytes: bytes,
    model_choice: str = "yolo26n",
    source_filename: str | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str | None = None,
):
    """
    MCP 接口：对单页漫画图像进行文本区域检测（Manga-Bubble-YOLO ONNX）。
    返回结构化检测结果：包含每个文本区域的 bbox/score/region_id/page_index。
    """
    return detect_text_regions(
        page_index=page_index,
        image_bytes=image_bytes,
        model_choice=model_choice,
        source_filename=source_filename,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")