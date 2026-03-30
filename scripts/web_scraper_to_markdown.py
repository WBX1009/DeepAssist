import os
import sys
import re
import traceback
from pathlib import Path

# 浏览器伪装
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

import nest_asyncio
nest_asyncio.apply()

import trafilatura
import ftfy
from slugify import slugify
from bs4 import BeautifulSoup, Comment, ProcessingInstruction, Doctype

from langchain_community.document_loaders.sitemap import SitemapLoader

# 🔥 核心：结构解析
from unstructured.partition.html import partition_html

# 项目路径
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 日志
from backend.core.logger import get_logger
logger = get_logger("GoldenDataCrawler")


class UltimateTechDocCrawler:
    def __init__(self, output_base_dir=None):
        self.output_base_dir = Path(
            output_base_dir or Path(project_root) / "data" / "raw_docs"
        )
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def _url_to_filename(self, url: str) -> str:
        return slugify(url, max_length=150) + ".md"

    # =============================
    # 🔥 HTML清洗（彻底修复 Unstructured lxml 报错）
    # =============================
    @staticmethod
    def clean_html(raw_html: str) -> str:
        # 1. 暴力消除 XML 处理指令 (直接解决 lxml 崩溃的罪魁祸首)
        raw_html = re.sub(r'<\?.*?\?>', '', raw_html, flags=re.DOTALL)

        # 2. 使用 html.parser，它比 lxml 更宽容
        soup = BeautifulSoup(raw_html, "html.parser")

        # 3. 物理拔除与正文无关的 UI 标签
        for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "svg", "form", "button"]):
            tag.decompose()

        # 4. 深度拔除 DOM 树里的不可见节点 (注释、残留指令等)
        for node in soup.find_all(string=True):
            if isinstance(node, (Comment, ProcessingInstruction, Doctype)):
                node.extract()

        return str(soup)

    # =============================
    # 🔥 智能质量过滤（不误杀标题和表格）
    # =============================
    @staticmethod
    def is_high_quality(text: str, element_type: str = "Text") -> bool:
        text = text.strip()
        if not text:
            return False

        # 如果是标题、代码或列表，绝对保留，无视长度校验！(修复误杀 Bug)
        if element_type in ["Title", "Code", "ListItem", "Table"]:
            return True

        # 对于普通纯文本，如果只有一两个单词且完全匹配 UI 词汇，则过滤
        lower_text = text.lower()
        ui_exact_matches =["previous", "next", "edit this page", "on this page", "search"]
        if lower_text in ui_exact_matches:
            return False

        # 广告和免责声明过滤
        noise_patterns =["all rights reserved", "subscribe to our newsletter"]
        if any(p in lower_text for p in noise_patterns):
            return False

        return True

    # =============================
    # 🔥 主处理函数（解析 -> 提纯 -> Markdown）
    # =============================
    def html_to_clean_markdown(self, raw_html: str) -> str:
        if not raw_html or len(raw_html) < 200:
            return ""

        try:
            # Step 0：洗去毒素，准备无菌 HTML
            clean_html_str = self.clean_html(raw_html)

            # Step 1：优先使用 Unstructured 进行结构化解析
            try:
                elements = partition_html(text=clean_html_str)
                blocks =[]

                for el in elements:
                    category = el.category # 提取到的元素类型：Title, Text, Code, Table 等
                    text = str(el).strip()

                    if not self.is_high_quality(text, category):
                        continue

                    # 🎯 将 Unstructured 元素还原为高质量 Markdown 格式
                    if category == "Title":
                        blocks.append(f"\n## {text}\n")
                    elif category == "ListItem":
                        blocks.append(f"- {text}")
                    elif category == "Code":
                        lines = text.split("\n")
                        if len(lines) > 50:
                            text = "\n".join(lines[:50]) + "\n\n...[长代码块已截断以节省 Context]..."
                        blocks.append(f"\n```text\n{text}\n```\n")
                    else:
                        blocks.append(text)

                final_md = "\n".join(blocks)

            except Exception as e:
                logger.warning(f"Unstructured解析失败，切换 Trafilatura 引擎: {e}")
                
                # Step 2：Fallback 到 Trafilatura
                extracted = trafilatura.extract(
                    clean_html_str,
                    include_formatting=True,
                    include_links=False
                )
                if not extracted:
                    return ""

                # 逐行弱过滤
                lines = ftfy.fix_text(extracted).split("\n")
                clean_lines =[]
                for line in lines:
                    if self.is_high_quality(line, "Text"):
                        clean_lines.append(line)
                final_md = "\n".join(clean_lines)

            # 🔥 文档级过滤：如果整篇文章洗完后连 50 个词都没有，说明是废页
            if len(final_md.split()) < 50:
                return ""

            # 压缩多余的空行
            final_md = re.sub(r"\n{3,}", "\n\n", final_md)
            return final_md.strip()

        except Exception as e:
            logger.warning(f"页面综合清洗失败: {e}")
            return ""

    # =============================
    # 🚀 主爬虫编排器
    # =============================
    def crawl_and_clean(self, category_name, sitemap_url, filter_urls=None, max_req=5):
        logger.info(f"🚀 开始抓取: {category_name}")
        category_dir = self.output_base_dir / category_name
        category_dir.mkdir(exist_ok=True)

        loader = SitemapLoader(
            web_path=sitemap_url,
            filter_urls=filter_urls,
            continue_on_failure=True,
            parsing_function=lambda soup: str(soup) if soup else ""
        )
        loader.requests_per_second = max_req

        try:
            docs = loader.load()
            success = 0

            for doc in docs:
                metadata = getattr(doc, "metadata", {}) or {}
                url = metadata.get("source", f"unknown_{success}")

                raw_html = getattr(doc, "page_content", "")
                final_md = self.html_to_clean_markdown(raw_html)

                if not final_md:
                    continue

                filename = self._url_to_filename(url)
                filepath = category_dir / filename

                header = f"---\nsource: {url}\ncategory: {category_name}\n---\n\n"

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(header + final_md)
                success += 1

            logger.info(f"✅ 完成 {category_name}：成功提纯 {success} 个高质量 Markdown 资产")

        except Exception:
            logger.error(traceback.format_exc())


# =============================
# 🎯 执行入口
# =============================
if __name__ == "__main__":
    crawler = UltimateTechDocCrawler()

    crawler.crawl_and_clean(
        "langchain",
        "https://docs.langchain.com/sitemap.xml",
        [r"https://docs\.langchain\.com/oss/python/"],
    )

    crawler.crawl_and_clean(
        "fastapi",
        "https://fastapi.tiangolo.com/sitemap.xml",[r"https://fastapi\.tiangolo\.com/.*"],
    )

    crawler.crawl_and_clean(
        "pydantic",
        "https://docs.pydantic.dev/latest/sitemap.xml",[r"https://docs\.pydantic\.dev/latest/.*"],
    )

    print("\n🎉 全部知识库构建完成！")