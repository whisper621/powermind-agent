#!/usr/bin/env python3
"""
PowerMind 通用知识库文档导入器 — 支持 .txt/.pdf/.docx/.md 自动拆分导入。

核心功能:
  1. 文件类型自动检测
  2. RecursiveCharacterTextSplitter 智能拆分
  3. metadata 自动标注 (data_type, province, date)
  4. MD5 去重 (避免重复导入)
  5. 批量写入 ChromaDB

设计原理:
  - 不是随意拆分 → 基于 RecursiveCharacterTextSplitter,按自然段落/句子边界拆分
  - 不是基于标题拆分 → 更稳健,兼容各种文档格式
  - 不是分集合存储 → 统一存储在 power_market_kb,通过 metadata.data_type 区分
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentImporter:
    """通用知识库文档导入器"""

    # 文档类型自动检测规则
    TYPE_DETECTION_RULES = {
        "policy_doc": {
            "keywords": ["政策", "规则", "办法", "通知", "规定", "条例", "细则",
                        "policy", "regulation", "rule", "notice", "市场交易", "偏差考核",
                        "现货交易", "中长期交易", "分时电价", "辅助服务"],
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        "expert_insight": {
            "keywords": ["复盘", "心得", "经验", "交易记录", "策略回顾",
                        "review", "lesson", "insight", "trading_log"],
            "chunk_size": 300,
            "chunk_overlap": 30,
        },
        "market_snapshot": {
            "keywords": ["行情", "市场报告", "日报", "周报", "月报", "快照",
                        "market_report", "daily", "weekly", "monthly", "snapshot"],
            "chunk_size": 400,
            "chunk_overlap": 40,
        },
    }

    def __init__(self, chroma_path: str | None = None):
        self._chroma_path = chroma_path or str(
            Path(__file__).resolve().parents[1] / "data" / "vectordb"
        )
        self._collection_name = "power_market_kb"
        self._md5_store_path = str(
            Path(__file__).resolve().parents[1] / "data" / "imported_md5.txt"
        )
        self._collection = None

    def _get_collection(self):
        """懒加载 ChromaDB collection"""
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=self._chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._collection = client.get_collection(self._collection_name)
        except Exception:
            self._collection = client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_splitter(self, chunk_size: int = 300, chunk_overlap: int = 30):
        """获取文本拆分器"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )

    def _load_file_content(self, file_path: str) -> str:
        """根据文件扩展名加载文件内容"""
        ext = Path(file_path).suffix.lower()

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                return "\n\n".join(d.page_content for d in docs)
            except ImportError:
                raise ImportError("需要安装 pypdf: pip install pypdf")

        elif ext in (".docx", ".doc"):
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                return "\n\n".join(d.page_content for d in docs)
            except ImportError:
                raise ImportError("需要安装 docx2txt: pip install docx2txt")

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _compute_md5(self, file_path: str) -> str:
        """计算文件 MD5"""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _is_imported(self, md5: str) -> bool:
        """检查 MD5 是否已导入"""
        store = Path(self._md5_store_path)
        if not store.exists():
            return False
        return md5 in store.read_text(encoding="utf-8").splitlines()

    def _mark_imported(self, md5: str):
        """标记 MD5 为已导入"""
        with open(self._md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5 + "\n")

    def _detect_data_type(self, filename: str) -> str:
        """基于文件名自动检测文档类型"""
        filename_lower = filename.lower()
        for dtype, rules in self.TYPE_DETECTION_RULES.items():
            for kw in rules["keywords"]:
                if kw.lower() in filename_lower:
                    return dtype
        return "general"

    def _extract_date_from_filename(self, filename: str) -> str:
        """尝试从文件名提取日期"""
        date_patterns = [
            r"(\d{4}[-_]\d{2}[-_]\d{2})",  # 2026-06-15
            r"(\d{4}年\d{1,2}月\d{1,2}日)",  # 2026年6月15日
            r"(\d{8})",  # 20260615
        ]
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        return datetime.now().strftime("%Y-%m-%d")

    def import_file(
        self,
        file_path: str,
        province: str = "全国",
        data_type: str = "auto",
        custom_metadata: dict | None = None,
    ) -> dict:
        """
        导入单个文件到知识库。

        Args:
            file_path: 文件路径
            province: 目标省份 (如 "山东", "全国")
            data_type: 文档类型 ("auto"=自动检测, "policy_doc", "expert_insight", "market_snapshot", "general")
            custom_metadata: 自定义元数据 (会合并到自动生成的元数据中)

        Returns:
            {"status": "success"/"skipped"/"error", "chunks": n, "file": filename}
        """
        filename = os.path.basename(file_path)
        logger.info(f"[DocumentImporter] 导入: {filename}")

        # 1. MD5 去重
        md5 = self._compute_md5(file_path)
        if self._is_imported(md5):
            logger.info(f"[DocumentImporter] 跳过 (已导入): {filename}")
            return {"status": "skipped", "chunks": 0, "file": filename, "reason": "duplicate"}

        # 2. 自动类型检测
        if data_type == "auto":
            data_type = self._detect_data_type(filename)

        # 3. 获取拆分参数
        type_config = self.TYPE_DETECTION_RULES.get(data_type, {"chunk_size": 300, "chunk_overlap": 30})
        chunk_size = type_config["chunk_size"]
        chunk_overlap = type_config["chunk_overlap"]

        # 4. 加载文件内容
        try:
            content = self._load_file_content(file_path)
        except Exception as e:
            logger.error(f"[DocumentImporter] 文件读取失败: {filename}: {e}")
            return {"status": "error", "chunks": 0, "file": filename, "reason": str(e)}

        if not content.strip():
            return {"status": "error", "chunks": 0, "file": filename, "reason": "空文件"}

        # 5. 拆分
        splitter = self._get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        from langchain_core.documents import Document
        docs = splitter.split_documents([Document(page_content=content, metadata={"source": file_path})])

        # 6. 构建 metadata
        base_metadata = {
            "source": file_path,
            "filename": filename,
            "data_type": data_type,
            "province": province,
            "import_date": datetime.now().strftime("%Y-%m-%d"),
            "file_md5": md5,
        }

        # 尝试从文件名提取日期
        extracted_date = self._extract_date_from_filename(filename)
        if extracted_date:
            base_metadata["doc_date"] = extracted_date

        if custom_metadata:
            base_metadata.update(custom_metadata)

        # 7. 写入 ChromaDB
        collection = self._get_collection()

        # 分批写入 (防止 ChromaDB 5461 限制)
        batch_size = 4000
        total_chunks = len(docs)
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for i, doc in enumerate(docs):
            chunk_id = f"{md5}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(doc.page_content)
            chunk_metadatas.append({**base_metadata, "chunk_index": i, "total_chunks": total_chunks})

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            try:
                collection.add(
                    ids=chunk_ids[batch_start:batch_end],
                    documents=chunk_texts[batch_start:batch_end],
                    metadatas=chunk_metadatas[batch_start:batch_end],
                )
            except Exception as e:
                logger.error(f"[DocumentImporter] ChromaDB 写入失败: batch {batch_start}-{batch_end}: {e}")
                return {"status": "error", "chunks": 0, "file": filename, "reason": str(e)}

        # 8. 标记已导入
        self._mark_imported(md5)

        logger.info(f"[DocumentImporter] 导入成功: {filename} → {total_chunks} chunks (type={data_type}, province={province})")
        return {
            "status": "success",
            "chunks": total_chunks,
            "file": filename,
            "data_type": data_type,
            "province": province,
            "md5": md5,
        }

    def import_directory(self, dir_path: str, province: str = "全国", recursive: bool = True) -> list[dict]:
        """
        批量导入目录下的所有支持文件。

        Args:
            dir_path: 目录路径
            province: 目标省份
            recursive: 是否递归子目录

        Returns:
            导入结果列表
        """
        supported_formats = {".txt", ".pdf", ".docx", ".doc", ".md"}
        results = []

        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"[DocumentImporter] 目录不存在: {dir_path}")
            return results

        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                result = self.import_file(str(file_path), province=province)
                results.append(result)
                print(f"  [{result['status']}] {result['file']} → {result.get('chunks', 0)} chunks")

        # 统计
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "error")
        print(f"\n导入完成: {success} 成功, {skipped} 跳过, {failed} 失败")

        return results

    def list_imported(self, data_type: str | None = None, province: str | None = None) -> list[dict]:
        """列出已导入的文档"""
        collection = self._get_collection()
        try:
            all_data = collection.get()
        except Exception:
            return []

        results = {}
        for meta in all_data.get("metadatas", []):
            key = meta.get("file_md5", "")
            if key and key not in results:
                # 按类型和省份过滤
                if data_type and meta.get("data_type") != data_type:
                    continue
                if province and meta.get("province") != province:
                    continue
                results[key] = {
                    "filename": meta.get("filename", ""),
                    "data_type": meta.get("data_type", ""),
                    "province": meta.get("province", ""),
                    "import_date": meta.get("import_date", ""),
                    "doc_date": meta.get("doc_date", ""),
                    "md5": key,
                }

        return list(results.values())


# ═══════════════════════════════════════════════════════════════════
# CLI 接口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind 知识库文档导入器")
    parser.add_argument("--file", "-f", help="导入单个文件")
    parser.add_argument("--dir", "-d", help="导入目录")
    parser.add_argument("--province", "-p", default="全国", help="省份")
    parser.add_argument("--type", "-t", default="auto", help="文档类型")
    parser.add_argument("--list", "-l", action="store_true", help="列出已导入文档")

    args = parser.parse_args()

    importer = DocumentImporter()

    if args.list:
        docs = importer.list_imported()
        if docs:
            print(f"\n已导入 {len(docs)} 个文档:\n")
            for d in docs:
                print(f"  [{d['data_type']}] {d['filename']} (省份: {d['province']}, 日期: {d['doc_date']})")
        else:
            print("知识库为空, 请先导入文档。")
        exit(0)

    if args.file:
        result = importer.import_file(args.file, province=args.province, data_type=args.type)
        print(f"\n导入结果: {result}")

    elif args.dir:
        results = importer.import_directory(args.dir, province=args.province)
        print(f"\n共处理 {len(results)} 个文件")

    else:
        parser.print_help()
        print("\n示例:")
        print("  python tools/document_importer.py --file 山东省现货交易规则.pdf --province 山东")
        print("  python tools/document_importer.py --dir ./my_policy_docs/ --province 山东")
        print("  python tools/document_importer.py --list")
