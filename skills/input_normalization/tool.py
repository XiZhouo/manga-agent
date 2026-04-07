from langchain_core.tools import tool
from skills.input_normalization.normalize import normalize_file

@tool
def normalize_input(file_bytes: bytes, file_type: str):
    """
    将用户上传的漫画文件规范化为按页图像序列。
    file_bytes: 文件原始 bytes
    file_type: 文件类型扩展名
    """
    return normalize_file(file_bytes, file_type)