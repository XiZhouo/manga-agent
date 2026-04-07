# skills/input_normalization/normalize.py

import io
import zipfile
from pathlib import Path
from typing import List, Dict, Union

from PIL import Image
from pdf2image import convert_from_bytes
import rarfile

# 定义数据结构
class NormalizedImage:
    def __init__(self, page_index: int, image_bytes: bytes, filename: str):
        self.page_index = page_index
        self.image_bytes = image_bytes
        self.filename = filename

def is_image_file(name: str) -> bool:
    """判断是否为图片文件"""
    n = (name or "").lower().strip()
    # allow both filename form ("a.png") and extension form ("png")
    if n in {"jpg", "jpeg", "png", "bmp"}:
        return True
    return n.endswith((".jpg", ".jpeg", ".png", ".bmp"))

def normalize_file(file_bytes: bytes, file_type: str) -> List[NormalizedImage]:
    """
    将任意输入文件转成图片序列输出。
    file_type 应该是后缀名，比如 "pdf"、"cbz"、"cbr"、"zip"、"jpg" 等
    """
    normalized: List[NormalizedImage] = []

    # 1. PDF 处理
    if file_type == "pdf":
        images = convert_from_bytes(file_bytes)
        for idx, img in enumerate(images):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            normalized.append(NormalizedImage(idx, buffer.getvalue(), f"page_{idx}.png"))
        return normalized

    # 2. Zip / Cbz 处理
    if file_type in ("zip", "cbz"):
        zip_ref = zipfile.ZipFile(io.BytesIO(file_bytes))
        # 取出所有图片并排序
        image_names = sorted([n for n in zip_ref.namelist() if is_image_file(n)])
        for idx, name in enumerate(image_names):
            img_bytes = zip_ref.read(name)
            normalized.append(NormalizedImage(idx, img_bytes, Path(name).name))
        return normalized

    # 3. Rar / Cbr 处理
    if file_type in ("rar", "cbr"):
        rar_ref = rarfile.RarFile(io.BytesIO(file_bytes))
        image_names = sorted([n for n in rar_ref.namelist() if is_image_file(n)])
        for idx, name in enumerate(image_names):
            img_bytes = rar_ref.read(name)
            normalized.append(NormalizedImage(idx, img_bytes, Path(name).name))
        return normalized

    # 4. 单图片
    if is_image_file(file_type):
        normalized.append(NormalizedImage(0, file_bytes, f"img_0.{file_type}"))
        return normalized

    # 5. 不支持类型
    raise ValueError(f"Unsupported file type: {file_type}")