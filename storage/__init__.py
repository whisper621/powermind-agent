"""
PowerMind 统一存储层 — 单目录、明确分工、自动标签。

数据流向:

  API 数据 (电价/天气/预测)  →  DataIngestionPipeline  →  SQL (直接入库，不保留源文件)
  人类上传 (政策/经验)        →  source_docs/           →  AutoImporter  →  ChromaDB
  实时流 (实时电价)           →  Redis 缓存 (不持久化)

多省份标签:
  - SQL: 所有表均有 province 列, WHERE province='山东'
  - ChromaDB: 所有文档均有 province metadata, where={"province":"山东"}
  - source_docs/: 按省份分子目录 (仅人类上传的文档)

目录:
  data/
  ├── source_docs/      ← 仅人类上传的文本文档 (政策/经验)
  ├── vectordb/         ← ChromaDB (代码自动维护)
  ├── sql/              ← SQLite (API 数据直接写入)
  ├── configs/          ← YAML 配置文件
  ├── templates/        ← 输出模板
  └── logs/             ← 审计/记忆日志
"""

from storage.auto_importer import AutoImporter, ensure_knowledge_ready
from storage.data_ingestion import DataIngestionPipeline, get_ingestion_pipeline
from storage.experience_archiver import ExperienceArchiver, get_experience_archiver

__all__ = [
    "AutoImporter", "ensure_knowledge_ready",
    "DataIngestionPipeline", "get_ingestion_pipeline",
    "ExperienceArchiver", "get_experience_archiver",
]
