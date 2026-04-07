from fastmcp import FastMCP
from skills.input_normalization.normalize import normalize_file

mcp = FastMCP("InputNormalization")

@mcp.tool()
def normalize(file_bytes: bytes, file_type: str):
    """MCP 接口：统一漫画输入格式."""
    return normalize_file(file_bytes, file_type)

if __name__ == "__main__":
    mcp.run(transport="stdio")