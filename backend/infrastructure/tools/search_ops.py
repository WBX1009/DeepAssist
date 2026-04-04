from duckduckgo_search import DDGS
from backend.core.logger import get_logger

logger = get_logger(__name__)

def web_search(query: str, max_results: int = 5) -> str:
    """
    使用搜索引擎在互联网上查询最新信息。
    当本地知识库查不到，或者需要获取实时资讯（如新闻、最新技术文档、现实世界事件）时调用此工具。
    :param query: 搜索关键词
    :param max_results: 返回的最大结果数量，默认 5
    """
    logger.info(f"🛠️ [Tool] 正在全网搜索: {query}")
    try:
        results =[]
        # DDGS 是一个免 API Key 的轻量级搜索库
        with DDGS() as ddgs:
            # 执行文本搜索
            search_results = ddgs.text(query, max_results=max_results)
            for idx, r in enumerate(search_results):
                title = r.get("title", "无标题")
                body = r.get("body", "无摘要")
                href = r.get("href", "")
                results.append(f"[{idx+1}] {title}\n摘要: {body}\n链接: {href}")
                
        if not results:
            return f"未能在互联网上搜索到关于 '{query}' 的有效信息。"
            
        return "互联网搜索结果如下：\n\n" + "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"全网搜索异常: {e}")
        return f"搜索引擎暂时不可用或触发了反爬限制: {str(e)}"