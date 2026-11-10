#!/usr/bin/env python3
"""
PowerMind 自动导入器 — 仅处理 source_docs/ 下人类上传的文本文档。

API 获取的结构化数据 (电价/天气/预测) → DataIngestionPipeline 直接写 SQL
人类上传的文本 (政策/经验)         → source_docs/ → 本模块 → ChromaDB

原则:
  - 数值数据不留源文件, 直接入库 (避免冗余)
  - 文本数据保留源文件 (审计追溯) + 写入 ChromaDB (向量检索)
  - 多省份通过 province metadata 区分
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


class AutoImporter:
    """Agent 启动时自动导入 source_docs/ 下的新文本文件到 ChromaDB"""

    def __init__(self):
        self._md5_store = DATA_ROOT / "logs" / "imported_md5.txt"
        self._chroma_collection = None

    def ensure_ready(self) -> dict:
        """确保知识库就绪: 扫描 source_docs/ 并导入新文档"""
        self._ensure_dirs()
        result = {"imported": 0, "skipped": 0, "errors": 0}

        src = DATA_ROOT / "source_docs"
        if src.exists():
            for f in src.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".txt", ".md", ".pdf", ".docx"):
                    r = self._import_document(f)
                    result[r["status"]] = result.get(r["status"], 0) + 1

        logger.info(f"[AutoImporter] 完成: 导入{result.get('imported',0)} 跳过{result.get('skipped',0)}")
        return result

    def _ensure_dirs(self):
        for d in ["source_docs", "source_docs/policies", "source_docs/experiences",
                   "vectordb", "sql", "configs", "templates", "logs"]:
            (DATA_ROOT / d).mkdir(parents=True, exist_ok=True)

    def _compute_md5(self, filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _is_imported(self, md5: str) -> bool:
        if not self._md5_store.exists():
            return False
        return md5 in self._md5_store.read_text(encoding="utf-8").splitlines()

    def _mark_imported(self, md5: str):
        self._md5_store.parent.mkdir(parents=True, exist_ok=True)
        with open(self._md5_store, "a", encoding="utf-8") as f:
            f.write(md5 + "\n")

    def _extract_metadata(self, filepath: Path) -> dict:
        """从路径提取 province 和 data_type"""
        parts = str(filepath).lower()
        province = "全国"
        for p in ("山东", "河南", "江苏", "广东", "浙江", "山西", "河北", "内蒙古"):
            if p in str(filepath):
                province = p
                break

        data_type = "general"
        if "policies" in parts:
            data_type = "policy_doc"
        elif "experiences" in parts:
            data_type = "expert_insight"

        return {"province": province, "data_type": data_type}

    def _import_document(self, filepath: Path) -> dict:
        """导入单个文本文档 → ChromaDB"""
        md5 = self._compute_md5(filepath)
        if self._is_imported(md5):
            return {"status": "skipped", "file": filepath.name}

        meta = self._extract_metadata(filepath)

        # 加载内容
        ext = filepath.suffix.lower()
        if ext in (".txt", ".md"):
            content = filepath.read_text(encoding="utf-8")
        elif ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                docs = PyPDFLoader(str(filepath)).load()
                content = "\n\n".join(d.page_content for d in docs)
            except Exception:
                return {"status": "errors", "file": filepath.name}
        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                docs = Docx2txtLoader(str(filepath)).load()
                content = "\n\n".join(d.page_content for d in docs)
            except Exception:
                return {"status": "errors", "file": filepath.name}
        else:
            return {"status": "errors", "file": filepath.name}

        if not content.strip():
            return {"status": "errors", "file": filepath.name}

        # 拆分 (政策长文本用大chunk，经验用中chunk)
        chunk_size = 500 if meta["data_type"] == "policy_doc" else 300
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_size // 10,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        chunks = splitter.split_documents([Document(page_content=content)])

        # 写入 ChromaDB
        col = self._get_chroma_collection()
        batch_size = 4000
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [f"{md5}_chunk_{j}" for j in range(i, min(i + batch_size, len(chunks)))]
            texts = [d.page_content for d in batch]
            metas = [{
                **meta,
                "source": str(filepath), "filename": filepath.name,
                "chunk_index": j, "total_chunks": len(chunks),
                "import_date": datetime.now().strftime("%Y-%m-%d"),
            } for j, d in enumerate(batch, start=i)]
            try:
                col.upsert(ids=ids, documents=texts, metadatas=metas)
            except Exception:
                col.add(ids=ids, documents=texts, metadatas=metas)

        self._mark_imported(md5)
        logger.info(f"[AutoImporter] {filepath.name} → {len(chunks)} chunks ({meta['data_type']}, {meta['province']})")
        return {"status": "imported", "file": filepath.name, "chunks": len(chunks)}

    def _get_chroma_collection(self):
        if self._chroma_collection:
            return self._chroma_collection
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(DATA_ROOT / "vectordb"),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._chroma_collection = client.get_collection("power_market_kb")
        except Exception:
            self._chroma_collection = client.create_collection(
                name="power_market_kb", metadata={"hnsw:space": "cosine"},
            )
        return self._chroma_collection


def ensure_knowledge_ready() -> dict:
    """Agent 启动时调用: 导入 source_docs/ 下的新文档"""
    return AutoImporter().ensure_ready()
