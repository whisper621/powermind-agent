#!/usr/bin/env python3
"""
政策雷达自动爬虫 — 自动扫描官方政策网站，发现新政策 → 下载 → 入库。

无需任何 API Key，纯 HTTP 请求。
支持: 国家能源局 / 发改委 / 31省能监办
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = (
    Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
    / "knowledge_base"
    / "knowledge_base.db"
)
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")

# ═══════════════════════════════════════════════════════════════════
# 种子 URL 配置
# ═══════════════════════════════════════════════════════════════════

SEED_URLS = [
    # 国家级
    {
        "url": "https://www.nea.gov.cn/xwzx/nyyw.htm",
        "province": "全国",
        "source_name": "国家能源局",
        "category": "综合",
    },
    {
        "url": "https://www.ndrc.gov.cn/xxgk/zcfb/",
        "province": "全国",
        "source_name": "国家发改委",
        "category": "综合",
    },
    # 省级能监办 (已确认可访问的)
    {
        "url": "https://sdb.nea.gov.cn/dtyw/tzgg/",
        "province": "山东",
        "source_name": "山东能监办",
        "category": "省内政策",
    },
    {
        "url": "https://jsb.nea.gov.cn/dtyw/tzgg/",
        "province": "江苏",
        "source_name": "江苏能监办",
        "category": "省内政策",
    },
    {
        "url": "https://zjb.nea.gov.cn/dtyw/tzgg/",
        "province": "浙江",
        "source_name": "浙江能监办",
        "category": "省内政策",
    },
    {
        "url": "https://gdb.nea.gov.cn/dtyw/tzgg/",
        "province": "广东",
        "source_name": "广东能监办",
        "category": "省内政策",
    },
    # 更多省份可按需添加
]

# 政策关键词匹配
POLICY_KEYWORDS = [
    "电力市场", "电力交易", "现货市场", "中长期", "分时电价",
    "偏差考核", "辅助服务", "新能源", "储能", "绿电",
    "售电", "入市", "申报", "结算", "容量电价",
    r"发改.*体改", r"国能.*发",
]

KEYWORD_PATTERN = re.compile("|".join(POLICY_KEYWORDS))

# ═══════════════════════════════════════════════════════════════════
# HTTP 工具
# ═══════════════════════════════════════════════════════════════════


def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    """HTTP GET 请求，返回解码后的文本。"""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 PowerMind-Agent/1.0 "
                "(power market policy crawler)"
            )
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # 尝试多种编码
            for enc in ["utf-8", "gb2312", "gbk", "gb18030"]:
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError) as e:
        print(f"  ⚠️ fetch error: {url} → {e}")
        return None


def extract_text_from_html(html: str) -> str:
    """从 HTML 提取纯文本。"""
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def extract_policy_links(
    html: str, base_url: str, province: str
) -> list[dict]:
    """从 HTML 中提取与电力市场相关的政策链接。"""
    links = []
    # 匹配 <a href="...">标题</a>
    pattern = re.compile(
        r'<a\s+[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>'
        r"([^<]{10,200})</a>",
        re.IGNORECASE,
    )
    for m in pattern.finditer(html):
        href = m.group(1)
        title = m.group(2).strip()
        title_clean = extract_text_from_html(title)

        if not KEYWORD_PATTERN.search(title_clean):
            continue

        # 构造完整 URL
        if href.startswith("http"):
            full_url = href
        else:
            full_url = urljoin(base_url, href)

        links.append({
            "title": title_clean[:120],
            "url": full_url,
            "province": province,
        })

    return links


# ═══════════════════════════════════════════════════════════════════
# 存储层
# ═══════════════════════════════════════════════════════════════════


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def already_saved(url: str, db_conn: sqlite3.Connection) -> bool:
    """检查 URL 是否已入库。"""
    row = db_conn.execute(
        "SELECT id FROM policy_docs WHERE source_url=?", (url,)
    ).fetchone()
    return row is not None


def save_to_sqlite(
    conn: sqlite3.Connection,
    title: str,
    province: str,
    category: str,
    content: str,
    url: str,
) -> int:
    """写入 SQLite policy_docs 表。"""
    # Ensure table exists
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS policy_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, province TEXT, province_id INTEGER, category TEXT,
            file_path TEXT, content TEXT, char_count INTEGER, source_url TEXT,
            doc_number TEXT, imported_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_policy_prov ON policy_docs(province);
        CREATE INDEX IF NOT EXISTS idx_policy_cat ON policy_docs(category);
    """)

    # Get province id
    pid = None
    row = conn.execute("SELECT id FROM provinces WHERE name=?", (province,)).fetchone()
    if row:
        pid = row[0]

    # Extract doc number
    doc_num = ""
    doc_match = re.search(
        r"[〔［]\d{4}[〕］]\d+号|发改.*?〔\d{4}〕\d+号|国能.*?〔\d{4}〕\d+号",
        content,
    )
    if doc_match:
        doc_num = doc_match.group(0)

    conn.execute(
        """INSERT OR IGNORE INTO policy_docs
           (title, province, province_id, category, content, char_count, source_url, doc_number)
           VALUES (?,?,?,?,?,?,?,?)""",
        (title[:200], province, pid, category, content[:10000], len(content), url, doc_num),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def save_to_chroma(
    doc_id: str,
    title: str,
    content: str,
    province: str,
    category: str,
) -> None:
    """写入 ChromaDB 向量库。"""
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection("power_market_kb")
        except Exception:
            collection = client.create_collection(
                name="power_market_kb",
                metadata={"hnsw:space": "cosine"},
            )

        collection.add(
            ids=[doc_id],
            documents=[f"{title}\n{content[:3000]}"],
            metadatas=[{
                "source": "policy_crawler",
                "data_type": "policy_doc",
                "province": province,
                "category": category,
                "title": title[:100],
            }],
        )
        print(f"    → ChromaDB indexed: {doc_id}")
    except Exception as e:
        print(f"    ⚠️ ChromaDB write error: {e}")


# ═══════════════════════════════════════════════════════════════════
# 主导入流程
# ═══════════════════════════════════════════════════════════════════


def detect_category(title: str, content: str) -> str:
    """根据标题和内容自动分类。"""
    text = title + content[:500]
    cat_map = {
        "现货": ["现货", "日前", "实时", "出清"],
        "中长期": ["中长期", "双边", "竞价", "挂牌", "年度交易"],
        "新能源": ["新能源", "风电", "光伏", "分布式", "可再生能源"],
        "辅助服务": ["辅助服务", "调峰", "调频", "备用", "储能"],
        "市场运行": ["电力市场", "交易规则", "市场规则", "市场建设"],
        "偏差考核": ["偏差", "考核", "免考", "结算"],
        "分时电价": ["分时", "峰谷", "尖峰", "低谷", "深谷"],
        "售电": ["售电", "履约", "保函"],
    }
    for cat, keywords in cat_map.items():
        if any(kw in text for kw in keywords):
            return cat
    return "综合"


def crawl_single_source(source: dict, db_conn: sqlite3.Connection) -> int:
    """扫描单个政策源。返回新发现数量。"""
    print(f"\n📡 扫描: {source['source_name']} ({source['province']})")
    html = fetch_url(source["url"])
    if not html:
        return 0

    links = extract_policy_links(html, source["url"], source["province"])
    print(f"  发现 {len(links)} 条电力相关链接")

    new_count = 0
    for link in links[:10]:  # 每次最多处理10条
        if already_saved(link["url"], db_conn):
            continue

        print(f"  📥 新政策: {link['title'][:60]}...")
        content = fetch_url(link["url"])
        if not content:
            continue

        text = extract_text_from_html(content)
        if len(text) < 200:
            continue

        category = detect_category(link["title"], text)

        # 双写入
        doc_id_val = save_to_sqlite(
            db_conn, link["title"], link["province"], category, text, link["url"]
        )
        save_to_chroma(
            f"policy-crawl-{doc_id_val}",
            link["title"],
            text,
            link["province"],
            category,
        )
        new_count += 1
        time.sleep(1.0)  # 礼貌爬取

    return new_count


def crawl_all() -> dict:
    """扫描所有种子源。"""
    print("=" * 60)
    print("  PowerMind Policy Crawler — 政策雷达")
    print(f"  启动时间: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  种子源数: {len(SEED_URLS)}")
    print("=" * 60)

    if not DB_PATH.exists():
        print("  ⚠️ SQLite 数据库不存在，跳过 SQLite 写入")
        return {"error": "db_not_found"}

    conn = sqlite3.connect(str(DB_PATH))
    total_new = 0
    results = []

    for source in SEED_URLS:
        try:
            n = crawl_single_source(source, conn)
            total_new += n
            results.append({"source": source["source_name"], "new": n})
        except Exception as e:
            print(f"  ❌ {source['source_name']}: {e}")
            results.append({"source": source["source_name"], "new": 0, "error": str(e)})

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"  扫描完成: 发现 {total_new} 条新政策")
    print(f"{'=' * 60}")

    return {
        "timestamp": datetime.now().isoformat(),
        "sources_scanned": len(SEED_URLS),
        "total_new_policies": total_new,
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind 政策雷达爬虫")
    parser.add_argument("--dry-run", action="store_true", help="只扫描，不入库")
    parser.add_argument("--source", help="仅扫描指定源名称")
    args = parser.parse_args()

    if args.source:
        source = next((s for s in SEED_URLS if s["source_name"] == args.source), None)
        if not source:
            print(f"未找到源: {args.source}")
            sys.exit(1)
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            n = crawl_single_source(source, conn)
            conn.close()
        else:
            print("DB not found, dry-run only")
    else:
        result = crawl_all()
        print(json.dumps(result, ensure_ascii=False, indent=2))
