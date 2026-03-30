import fitz  # PyMuPDF
import re
import os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_FILE_PATH = PROJECT_ROOT / "大模型应用开发八股文.pdf"
OUTPUT_MD_DIR = PROJECT_ROOT / "data" / "raw_docs"
OUTPUT_MD_PATH = OUTPUT_MD_DIR / "llm_interview_cleaned.md"

def merge_semantic_lines(raw_lines):
    """核心算法：语义缝合引擎"""
    merged_lines =[]
    current_paragraph = ""

    for line in raw_lines:
        line = line.strip()
        if not line: continue

        if re.match(r'^#|^[0-9]+\.\d|^\-|^\*\*', line) or re.match(r'^(解读|知识点|答案|步骤\w*|结论|拓展思考)[:：]', line):
            if current_paragraph:
                merged_lines.append(current_paragraph)
            current_paragraph = line
            continue

        if current_paragraph and re.search(r'[。？！：；:;]$', current_paragraph):
            merged_lines.append(current_paragraph)
            current_paragraph = line
            continue

        if current_paragraph and re.search(r'[a-zA-Z\-]$', current_paragraph) and re.match(r'^[a-zA-Z]', line):
            current_paragraph += line
            continue

        if current_paragraph:
            if re.search(r'[a-zA-Z0-9]$', current_paragraph) and re.match(r'^[a-zA-Z0-9]', line):
                current_paragraph += " " + line
            else:
                current_paragraph += line
        else:
            current_paragraph = line

    if current_paragraph:
        merged_lines.append(current_paragraph)
    return merged_lines

def post_process_formatting(lines):
    """最终格式化美化引擎"""
    formatted_lines =[]
    for i, line in enumerate(lines):
        if re.match(r'^#|^\*\*', line):
            if i > 0 and formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
        formatted_lines.append(line)

    final_output =[]
    for i, line in enumerate(formatted_lines):
        if line == "" and i > 0 and final_output and final_output[-1] == "":
            continue
        final_output.append(line)
    return final_output

# ... (前面的 merge_semantic_lines 等函数保持不变) ...

def process_pdf_to_md_ultimate(pdf_path: Path, output_path: Path):
    doc = fitz.open(pdf_path)
    raw_lines = []
    for page in tqdm(doc, desc="Extracting"):
        # 裁剪 40 像素避开页眉页脚
        clip = fitz.Rect(0, 40, page.rect.width, page.rect.height - 40)
        text = page.get_text("text", clip=clip)
        if text: raw_lines.extend(text.split('\n'))
    doc.close()

    # 1. 清洗掉干扰符
    clean_lines = []
    in_main_content = False
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        line = re.sub(r'@@@[\d\.\s]+@@@', '', line)
        # 定位正文起点
        if not in_main_content:
            if re.match(r'^1\s+模型选型与业务需求匹配', line): in_main_content = True
            else: continue
        clean_lines.append(line)

    # 2. 语义缝合
    merged_lines = merge_semantic_lines(clean_lines)

    # 3. 结构化处理 (核心逻辑修复)
    structured_lines = []
    
    # 状态位：是否处于“答案/内容区”
    # 在这个区域内，所有的 1.1, 1.2 都被视为正文，直到遇到下一个 ### 或 ## 或 #
    is_content_zone = False 

    for line in merged_lines:
        # 优先判断是否是 3 级标题（具体的面试题）
        if re.match(r'^\d+\.\d+\.\d+\s', line):
            structured_lines.append(f"\n### {line}\n")
            is_content_zone = True # 开启内容保护模式
            continue

        # 判断是否是 1 级或 2 级标题
        # 只有在非内容区，或者满足特定的“大章节”特征时才识别
        is_h2 = re.match(r'^\d+\.\d+\s', line)
        is_h1 = re.match(r'^\d+\s(?!\.)', line)

        if (is_h1 or is_h2) and is_content_zone:
            # 这里的逻辑是：如果在问题内部看到了类似 2.1 的数字
            # 我们需要通过它是否紧跟着“正文模块”来判断它是不是新章节
            # 面试题库特征：新章节通常很短。
            if len(line) < 60: # 章节名通常不会太长
                is_content_zone = False # 退出内容模式，识别为新标题
            else:
                # 依然保持为正文
                pass 

        if not is_content_zone:
            if is_h2:
                structured_lines.append(f"\n## {line}\n")
                continue
            if is_h1:
                structured_lines.append(f"\n# {line}\n")
                continue
        
        # 处理正文内容块
        # 将解读、知识点等模块加粗
        line = re.sub(r'^(解读|知识点|答案|结论|拓展思考)[:：]', r'**\1:** ', line)
        structured_lines.append(line)

    # 4. 写入文件（保持紧凑的换行）
    OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # 用单个换行符连接，保证正文紧凑
        f.write('\n'.join(structured_lines))

    print(f"✅ ETL 成功！已修复标题误判问题。")

if __name__ == "__main__":
    process_pdf_to_md_ultimate(PDF_FILE_PATH, OUTPUT_MD_PATH)