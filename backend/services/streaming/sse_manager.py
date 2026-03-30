import json

class SSEManager:
    """SSE 流式协议封装工具"""
    
    @staticmethod
    def format_chunk(text: str) -> str:
        """
        将文本块包装为标准 SSE 格式: `data: {"content": "..."}\n\n`
        使用 JSON 序列化是为了安全地转义换行符和引号。
        """
        payload = json.dumps({"content": text}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    @staticmethod
    def format_end() -> str:
        """发送结束信号"""
        return "data:[DONE]\n\n"
        
    @staticmethod
    def format_error(err_msg: str) -> str:
        """发送错误信号"""
        payload = json.dumps({"error": err_msg}, ensure_ascii=False)
        return f"data: {payload}\n\n"