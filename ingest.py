from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOC_DIR, CHUNK_SIZE, CHUNK_OVERLAP, logger
from vectorstore import get_store


def load_text(path: str | Path) -> str:
   
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".pdf",):
        return _parse_pdf(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".csv",):
        return _parse_csv(path)
    raise ValueError(f"暂不支持的文件类型: {suffix}（支持 PDF/TXT/CSV）")


def _parse_pdf(path: Path) -> str:
    # 优先使用 unstructured.partition_pdf（作业要求方法），提取数据元素
    text = ""
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=str(path), strategy="fast")
        text = "\n".join(e.text for e in elements if getattr(e, "text", None))
        logger.info("ingest: 使用 unstructured 解析 %s（%d 元素）", path.name, len(elements))
    except Exception as e:
        logger.info("ingest: unstructured 解析失败（%s），回退 pypdf", e)

    if text and text.strip():
        return text

    # 兜底：unstructured 未提取到文本时，改用 pypdf（保证 PDF 解析入库可用）
    logger.info("ingest: partition_pdf 未提取到文本，改用 pypdf 解析 %s", path.name)
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _parse_csv(path: Path) -> str:
    import csv
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            rows.append(" | ".join(cell.strip() for cell in row if cell))
    return "\n".join(rows)


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_file(path: str | Path, store_name: str = "default",
                metadata: Optional[dict] = None) -> int:
    path = Path(path)
    text = load_text(path)
    chunks = split_text(text)
    metas = [
        {"source": str(path), **({"category": metadata.get("category")}
                                  if metadata and metadata.get("category") else {})}
        for _ in chunks
    ]
    store = get_store(store_name)
    store.add_texts(chunks, metadatas=metas)
    logger.info("ingest: %s → %s 已入库 %d 个 chunk（当前共 %d 条）",
                path.name, store_name, len(chunks), store.count())
    return len(chunks)


def ingest_directory(directory: str | Path = DOC_DIR, store_name: str = "default") -> int:
    directory = Path(directory)
    total = 0
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in (".pdf", ".txt", ".md", ".csv"):
            try:
                total += ingest_file(f, store_name=store_name)
            except Exception as e:
                logger.exception("ingest: 处理 %s 失败", f.name)
    logger.info("ingest: 目录批量入库完成，共 %d 个 chunk", total)
    return total


if __name__ == "__main__":
    import sys
    store_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    target = sys.argv[1] if len(sys.argv) > 1 else str(DOC_DIR)
    n = ingest_file(target, store_name) if Path(target).is_file() else \
        ingest_directory(target, store_name)
    print(f"入库完成，共 {n} 个 chunk（store={store_name}）")
