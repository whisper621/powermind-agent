#!/usr/bin/env python3
"""
PowerMind 模拟数据生成器 V3 — 生成售电交易侧测试所需的全部假数据。

数据存放遵循统一 data/ 目录规范:
  - 源文件 → data/source_docs/ (政策/经验) + data/source_data/ (结构化)
  - 向量库 → data/vectordb/ (ChromaDB 自动生成)
  - 数据库 → data/sql/ (SQLite 自动生成)
  - 配置 → data/configs/

★ 数据流向:
  人类放文件到 data/source_docs/ 和 data/source_data/
  → AutoImporter 启动时自动检测新文件
  → 导入到 data/vectordb/ (ChromaDB) + data/sql/ (SQLite)
  → Agent 查询 data/vectordb/ + data/sql/

使用方法:
  python scripts/generate_mock_data.py --all           # 仅生成文件
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
random.seed(42)


# ═══════════════════════════════════════════════════════════════════
# 1. 电价预测数据 (24点逐时 + P10/P50/P90) — 核心
# ═══════════════════════════════════════════════════════════════════

def generate_price_forecasts(days: int = 15, province: str = "山东"):
    """
    生成模拟【日前电价预测】数据。

    这是售电交易侧最核心的数据 —— 每个预测包含:
    - 24点逐时电价 (元/MWh)
    - P10/P50/P90 置信区间
    - 峰谷均价
    - 置信度
    - 市场特征标签

    存放位置: 01_structured_data/import_staging/{省份}/spot_prices/
    """
    staging = DATA / "source_data" / province / "spot_prices"
    staging.mkdir(parents=True, exist_ok=True)

    forecasts = []
    today = datetime.now()

    for d in range(days):
        target_date = today + timedelta(days=d)
        date_str = target_date.strftime("%Y-%m-%d")
        month = target_date.month

        # 季节基准
        if month in (6, 7, 8):
            base = 420
            seasonal_tag = "夏季"
        elif month in (12, 1, 2):
            base = 400
            seasonal_tag = "冬季"
        else:
            base = 360
            seasonal_tag = "春秋"

        hourly = []
        for hour in range(24):
            if 0 <= hour < 6:
                factor = 0.65   # 深谷
            elif 6 <= hour < 10:
                factor = 0.90   # 早峰
            elif 10 <= hour < 15:
                factor = 0.80 if month in (6,7,8) else 1.0  # 午间光伏
            elif 15 <= hour < 18:
                factor = 1.05   # 午后
            elif 18 <= hour < 21:
                factor = 1.35   # 晚峰
            else:
                factor = 0.85   # 夜间

            noise = random.gauss(0, 25)
            price = max(0, min(1500, base * factor + noise))
            hourly.append(round(price, 2))

        avg_price = round(sum(hourly) / 24, 2)
        peak_avg = round(sum(hourly[18:21]) / 3, 2)
        valley_avg = round(sum(hourly[0:6]) / 6, 2)
        spread = round(peak_avg - valley_avg, 2)

        # 置信区间
        p50 = hourly
        p10 = [round(max(0, p - random.uniform(20, 40)), 2) for p in hourly]
        p90 = [round(min(1500, p + random.uniform(20, 40)), 2) for p in hourly]

        # 确定综合方向和风险
        if avg_price > base * 1.1:
            direction, risk = "bullish", "medium"
        elif avg_price < base * 0.9:
            direction, risk = "bearish", "medium"
        else:
            direction, risk = "neutral", "low"

        # 随机极端日
        if random.random() < 0.05:
            risk = "high"
            if random.random() < 0.5:
                direction = "bullish"
                hourly = [round(p * 1.3, 2) for p in hourly]
            else:
                direction = "bearish"
                hourly = [round(p * 0.7, 2) for p in hourly]

        forecasts.append({
            "province": province,
            "target_date": date_str,
            "model_name": "company_v1",
            "model_version": "1.0",
            "avg_price": avg_price,
            "peak_avg": peak_avg,
            "valley_avg": valley_avg,
            "spread": spread,
            "confidence": round(0.75 + random.uniform(-0.15, 0.15), 2),
            "direction": direction,
            "risk": risk,
            "hourly_predictions": hourly,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "seasonal_tag": seasonal_tag,
        })

    # 保存为 JSON (供导入脚本使用)
    json_path = staging / f"{province}_forecasts_{forecasts[0]['target_date']}_to_{forecasts[-1]['target_date']}.json"
    json_path.write_text(json.dumps(forecasts, ensure_ascii=False, indent=2), encoding="utf-8")

    # 也保存为 CSV 便于查看
    csv_path = staging / f"{province}_forecasts_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target_date", "avg_price", "peak_avg", "valley_avg", "spread",
            "confidence", "direction", "risk", "seasonal_tag",
            "h00","h01","h02","h03","h04","h05","h06","h07",
            "h08","h09","h10","h11","h12","h13","h14","h15",
            "h16","h17","h18","h19","h20","h21","h22","h23",
        ])
        writer.writeheader()
        for f in forecasts:
            row = {
                "target_date": f["target_date"],
                "avg_price": f["avg_price"],
                "peak_avg": f["peak_avg"],
                "valley_avg": f["valley_avg"],
                "spread": f["spread"],
                "confidence": f["confidence"],
                "direction": f["direction"],
                "risk": f["risk"],
                "seasonal_tag": f["seasonal_tag"],
            }
            for i, p in enumerate(f["hourly_predictions"]):
                row[f"h{i:02d}"] = p
            writer.writerow(row)

    print(f"[假数据] 电价预测: {json_path} ({len(forecasts)} 天)")
    print(f"[假数据] 电价预测: {csv_path}")
    return str(json_path), forecasts


# ═══════════════════════════════════════════════════════════════════
# 2. 模拟政策文档 → 放入 02_text_knowledge/
# ═══════════════════════════════════════════════════════════════════

POLICY_DOCS = {
    "山东省电力现货市场交易规则_2025.txt": """山东省电力现货市场交易规则

第一章 总则
第一条 为规范山东省电力现货市场交易行为，保障市场公平、有序运行，制定本规则。

第二章 日前市场交易
第五条 日前市场采用全电量申报、集中优化出清的方式。
第六条 发电企业须在每个交易日12:00前提交次日发电报价曲线。
第七条 售电公司须在每个交易日12:00前提交次日购电需求申报。
第八条 日前市场出清价格上限为1500元/MWh，下限为0元/MWh。

第三章 实时市场交易
第九条 实时市场采用偏差电量出清方式。

第四章 中长期交易
第十一条 中长期交易包括年度、月度、月内滚动撮合等多种交易品种。

第五章 偏差考核
第十四条 售电公司月累计偏差电量在±5%以内的，免于偏差考核。
第十五条 超出±5%的部分，按日前出清均价的110%进行偏差结算。
第十六条 连续3个月偏差超过±10%的，暂停交易资格。""",

    "山东省分时电价政策_2025.txt": """山东省分时电价政策

一、分时时段划分
尖峰时段（2小时）：18:00-20:00（夏季6-8月、冬季12-2月）
峰段（6小时）：10:00-15:00, 18:00-21:00
平段（8小时）：07:00-10:00, 15:00-18:00, 21:00-23:00
谷段（8小时）：23:00-07:00
深谷时段（4小时）：00:00-06:00（仅在新能源消纳困难的春季和秋季启用）

二、分时电价浮动比例
- 尖峰电价 = 平段电价 × 1.8
- 峰段电价 = 平段电价 × 1.5
- 平段电价 = 基准电价
- 谷段电价 = 平段电价 × 0.5
- 深谷电价 = 平段电价 × 0.3

三、执行范围
适用于所有参与电力市场的工商业用户。""",

    "山东省电力中长期交易细则_2025.txt": """山东省电力中长期交易实施细则

一、交易品种
1. 年度双边协商交易：每年12月进行次年年度交易，合同期限1年。
2. 月度集中竞价交易：每月25日前进行次月交易，边际出清。
3. 月内滚动撮合交易：每月1日-25日每个工作日，上午10:00-11:30。

二、偏差结算
- 正偏差：超出部分按实时市场价格结算
- 负偏差：不足部分按合同价格的90%结算
- 偏差免考核范围：±5%

三、合同转让
- 发电侧合同转让：可在月度集中竞价市场转让
- 售电侧合同转让：可在月内滚动撮合市场转让，上限为原合同的50%

四、结算公式
总费用 = 合同电量结算 + 偏差电量结算 + 辅助服务费用""",
}


def generate_policy_docs(province: str = "山东"):
    """生成模拟政策文档 → 02_text_knowledge/policy_docs/"""
    output_dir = DATA / "source_docs" / "policies" / province / "2025"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in POLICY_DOCS.items():
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"[假数据] 政策文档: {filepath}")

    return str(output_dir)


# ═══════════════════════════════════════════════════════════════════
# 3. 模拟交易复盘经验 → 放入 02_text_knowledge/expert_logs/
# ═══════════════════════════════════════════════════════════════════

def generate_expert_logs(province: str = "山东"):
    """生成模拟交易复盘 → 02_text_knowledge/expert_logs/"""
    output_dir = DATA / "source_docs" / "experiences" / province
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = [
        {
            "date": "2025-08-12", "category": "日前申报",
            "title": "高温日午间光伏大发-抄底申报策略",
            "content": """高温日午间光伏大发，采取"早峰高价出货、午间低价保量"策略。
早峰报价385，午间降至280保量，晚峰恢复450溢价。
结果: 均价378，偏差在±5%免考核范围内。
教训: 高温日光伏出力预判是关键，晚峰溢价策略有效，需要更精准的光伏出力预测模型。""",
            "tags": "高温,光伏,晚峰,鸭子曲线,溢价",
            "direction": "bullish", "result": "success",
        },
        {
            "date": "2025-07-23", "category": "实时交易",
            "title": "省外来电突增导致午间价格暴跌-紧急应对",
            "content": """省外来电突增约3000MW，午间现货价格从320急跌至185。
实时补仓500MWh均价195，摊薄日平均成本，成功挽回约60%偏差损失。
教训: 必须持续监盘实时市场，建立省外来电异常监测机制，中长期锁价比例过低导致敞口过大。""",
            "tags": "省外来电,实时补仓,价格暴跌,偏差,风控",
            "direction": "bearish", "result": "partial_loss",
        },
        {
            "date": "2025-11-15", "category": "中长期锁价",
            "title": "冬季供暖前提前锁价-中长期策略成功案例",
            "content": """11月中旬以365元/MWh锁定12月总用电量60%(3000MWh)。
12月实际日前均价398，锁价部分节省电费约9.9万元。
剩余40%通过月内滚动撮合以均价385成交。
教训: 供暖季前提前1个月锁价是明智策略，60%锁价比例合理。""",
            "tags": "中长期,锁价,冬季,煤价,滚动撮合",
            "direction": "neutral", "result": "success",
        },
        {
            "date": "2025-09-05", "category": "日前申报",
            "title": "台风天气新能源出力骤降-看涨策略失误",
            "content": """台风导致风电停运、光伏出力几乎为零。判断供需偏紧，全天报价高15-20%。
实际出清价低于预期(均价328 vs 申报380)，偏差+15.8%触发考核。
教训: 台风对工业负荷的冲击超过对发电侧的影响，极端天气下不应过于激进。""",
            "tags": "台风,新能源,负荷下降,偏差考核,极端天气",
            "direction": "bullish", "result": "failure",
        },
        {
            "date": "2026-03-20", "category": "综合策略",
            "title": "春季新能源大发+负荷低谷-低价套利策略",
            "content": """春季周末风光出力高位+负荷低谷，供应严重过剩。
日前均价报250保量，午间低至200。储能深谷180充电、晚峰380放电，套利1850元。
月度锁价仅40%，现货部分成本低至238，综合降本显著。
教训: 春季周末是确定性低价窗口，低价日减少中长期增加现货敞口是正确的。""",
            "tags": "春季,新能源大发,低价,储能套利,中长期",
            "direction": "bearish", "result": "success",
        },
    ]

    # CSV
    csv_path = output_dir / f"{province}_expert_trading_logs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date","province","category","title","content","tags","direction","result"])
        writer.writeheader()
        for log in logs:
            log["province"] = province
            writer.writerow(log)

    # Markdown
    for log in logs:
        md_path = output_dir / f"{log['date']}_{log['title'][:25]}.md"
        md_content = f"""# {log['title']}
- 日期: {log['date']} | 省份: {province} | 分类: {log['category']}
- 标签: {log['tags']} | 方向: {log['direction']} | 结果: {log['result']}

{log['content']}
"""
        md_path.write_text(md_content, encoding="utf-8")

    print(f"[假数据] 交易复盘: {csv_path} ({len(logs)} 条)")
    return str(output_dir)


# ═══════════════════════════════════════════════════════════════════
# 4. 模拟历史行情快照 → 放入 01_structured_data/import_staging/
# ═══════════════════════════════════════════════════════════════════

def generate_market_snapshots(days: int = 30, province: str = "山东"):
    """生成模拟历史行情快照 → 结构化数据"""
    staging = DATA / "source_data" / province / "spot_prices"
    staging.mkdir(parents=True, exist_ok=True)

    snapshots = []
    today = datetime.now()

    for d in range(days, 0, -1):
        date_obj = today - timedelta(days=d)
        date_str = date_obj.strftime("%Y-%m-%d")
        month = date_obj.month

        if month in (6,7,8):
            base, wind_base, solar_base = 420, 2500, 6000
        elif month in (12,1,2):
            base, wind_base, solar_base = 400, 4000, 2000
        else:
            base, wind_base, solar_base = 360, 3500, 4500

        noise = random.gauss(0, 40)
        avg_price = max(50, min(800, base + noise))

        snapshots.append({
            "province": province,
            "date": date_str,
            "avg_price": round(avg_price, 1),
            "peak_price": round(avg_price * 1.35 + random.gauss(0, 20), 1),
            "valley_price": round(avg_price * 0.65 + random.gauss(0, 10), 1),
            "total_load_mw": round(base * 135 + random.gauss(0, 5000), 0),
            "wind_output_mw": round(wind_base + random.gauss(0, 500), 0),
            "solar_output_mw": round(solar_base * random.uniform(0.6, 1.4), 0),
            "temperature": round(25 + 10*(month in(6,7,8)) + random.gauss(0,3), 1),
            "precipitation": round(max(0, random.gauss(2, 5)), 1),
            "direction": "bullish" if avg_price > 380 else ("bearish" if avg_price < 320 else "neutral"),
            "risk": "low" if abs(noise) < 30 else ("high" if abs(noise) > 60 else "medium"),
        })

    json_path = staging / f"{province}_market_snapshots_30d.json"
    json_path.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[假数据] 行情快照: {json_path} ({len(snapshots)} 天)")
    return str(json_path), snapshots


# ═══════════════════════════════════════════════════════════════════
# 5. 全部生成 + 导入
# ═══════════════════════════════════════════════════════════════════

def generate_all(province: str = "山东"):
    """生成全部假数据到 powermind-knowledge-base/ 对应位置"""
    print("=" * 60)
    print("PowerMind 模拟数据生成器 V2")
    print(f"目标省份: {province}")
    print(f"知识库路径: {KB}")
    print("=" * 60)
    print()

    results = {}

    print("[1/4] 生成电价预测数据 (24点逐时 + P10/P50/P90)...")
    results["forecasts_path"], results["forecasts"] = generate_price_forecasts(province=province)

    print("[2/4] 生成政策文档...")
    results["policies_path"] = generate_policy_docs(province=province)

    print("[3/4] 生成交易复盘经验...")
    results["expert_logs_path"] = generate_expert_logs(province=province)

    print("[4/4] 生成历史行情快照...")
    results["snapshots_path"], results["snapshots"] = generate_market_snapshots(province=province)

    print()
    print("=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print()
    print("文件位置:")
    print(f"  电价预测: {results['forecasts_path']}")
    print(f"  政策文档: {results['policies_path']}")
    print(f"  交易复盘: {results['expert_logs_path']}")
    print(f"  行情快照: {results['snapshots_path']}")

    return results


def import_all_to_knowledge_base(province: str = "山东"):
    """生成假数据 + 自动导入到 ChromaDB + PostgreSQL"""

    # Step 1: 生成
    data = generate_all(province=province)

    print()
    print("=" * 60)
    print("导入到知识库...")
    print("=" * 60)

    # Step 2: 导入文本知识到 ChromaDB
    print("\n[导入 ChromaDB] 政策文档 + 交易复盘...")
    from tools.document_importer import DocumentImporter
    importer = DocumentImporter()

    importer.import_directory(str(DATA / "source_docs" / "policies" / province), province=province)
    importer.import_directory(str(DATA / "source_docs" / "experiences" / province), province=province)

    # Step 3: 导入预测数据到 ChromaDB + SQLite
    print("\n[导入] 电价预测归档...")
    from scripts.data_archiver import ForecastArchiver
    fa = ForecastArchiver()
    for f in data["forecasts"]:
        fa.archive_forecast(f)
    print(f"  归档了 {len(data['forecasts'])} 条电价预测")

    # Step 4: 导入行情快照
    print("\n[导入] 行情快照...")
    from scripts.data_archiver import MarketSnapshotArchiver
    msa = MarketSnapshotArchiver()
    for s in data["snapshots"]:
        msa.archive_snapshot(s)
    print(f"  归档了 {len(data['snapshots'])} 条行情快照")

    # Step 5: 验证
    print("\n" + "=" * 60)
    print("导入验证")
    print("=" * 60)
    docs = importer.list_imported()
    for d in docs:
        print(f"  [{d['data_type']}] {d['filename']} (省份: {d['province']})")
    print(f"\n总计: {len(docs)} 个文档")

    print()
    print("=" * 60)
    print("全部完成！接下来可以运行:")
    print("  python scripts/evaluate_rag.py")
    print("  chainlit run app_chainlit.py --port 8000")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind 模拟数据生成器 V2")
    parser.add_argument("--all", action="store_true", help="生成全部假数据文件")
    parser.add_argument("--import-all", action="store_true", help="生成 + 自动导入到知识库")
    parser.add_argument("--province", "-p", default="山东", help="目标省份")

    args = parser.parse_args()

    if args.import_all:
        import_all_to_knowledge_base(province=args.province)
    elif args.all:
        generate_all(province=args.province)
    else:
        parser.print_help()
        print("\n推荐:")
        print("  python scripts/generate_mock_data.py --import-all  # 一键搞定")
