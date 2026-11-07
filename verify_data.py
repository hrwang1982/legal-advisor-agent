"""法规库数据体检（零第三方依赖）。

目的：规则引擎的正确性依赖元数据完整性。元数据缺失或错误会让规则「永不触发」或「错误触发」——
这类问题在规则单测里发现不了（单测用的是合成数据），必须靠数据体检拦截。

运行：python3 verify_data.py     缺失或错误关键元数据即 exit 1（可接入 CI）
"""
import json
import os
import re
import sys

from app.ingest import build_version_timelines
from app.rules import BASE_RANK

DATA = "app/data/sample_laws.json"

VALID_HIERARCHIES = set(BASE_RANK) | {"司法解释"}
VALID_NORM_TYPES = {"substantive", "procedural", "judicial_interpretation", "administrative_penalty", ""}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 关键字段：缺失会让检索、位阶裁决或时间效力判定失效
REQUIRED_NONEMPTY = [
    "law_name", "hierarchy", "article", "content", "source", "category", "validity",
]
# 日期字段：允许为空，但非空必须格式合法
DATE_FIELDS = [
    "promulgation_date", "effective_date", "article_effective_date", "effective_to", "host_effective_date",
]


def main() -> int:
    rows = []
    try:
        from app.mysql_store import mysql_store
        db_rows = mysql_store.query("SELECT * FROM sample_laws")
        if db_rows:
            for r in db_rows:
                meta = {}
                if r.get("metadata_json"):
                    try:
                        meta = json.loads(r["metadata_json"])
                    except Exception:
                        pass
                item = dict(meta) if meta else dict(r)
                rows.append(item)
    except Exception:
        pass

    if not rows and os.path.exists(DATA):
        with open(DATA, "r", encoding="utf-8") as f:
            rows = json.load(f)

    errors, warns = [], []
    seen_keys = set()

    for i, r in enumerate(rows):
        tag = f"[{i}] {r.get('law_name', '?')}{r.get('article', '')}{r.get('paragraph', '')}"

        # 0) 重复条文拦截（同一部法 + 同一条号 + 同一款项不得重复入库）
        key = (r.get("law_name", ""), r.get("article", ""), r.get("paragraph", ""))
        if key in seen_keys:
            errors.append(f"{tag}：存在重复入库条文 → {key}")
        seen_keys.add(key)

        for k in REQUIRED_NONEMPTY:
            if not str(r.get(k, "")).strip():
                errors.append(f"{tag}：必填字段为空 → {k}")

        for k in DATE_FIELDS:
            v = str(r.get(k, "") or "").strip()
            if v and not DATE_RE.match(v):
                errors.append(f"{tag}：日期格式非法 → {k}={v}")

        h = r.get("hierarchy", "")
        if h not in VALID_HIERARCHIES:
            errors.append(f"{tag}：非法位阶 → {h}")

        nt = str(r.get("norm_type", "") or "").strip()
        if nt not in VALID_NORM_TYPES:
            errors.append(f"{tag}：非法 norm_type → {nt}")

        # 民法典等全国人大通过的基本法律，制定机关不得错标为常委会
        if r.get("law_name") == "中华人民共和国民法典" and r.get("source") != "全国人民代表大会":
            errors.append(f"{tag}：《民法典》制定机关应为「全国人民代表大会」，当前为 {r.get('source')!r}")

        # 司法解释必须声明被解释法律，否则位阶与时间效力边界无法依附
        if h == "司法解释":
            if not str(r.get("host_law", "")).strip():
                errors.append(f"{tag}：司法解释缺少 host_law，位阶无法依附")
            if str(r.get("host_hierarchy", "")).strip() not in BASE_RANK:
                errors.append(f"{tag}：司法解释 host_hierarchy 缺失或非法 → {r.get('host_hierarchy')!r}")

        # 失效条文必须能追溯到替代依据或失效日，否则无法做时间定位
        if r.get("validity") not in ("现行有效", "", None):
            if not str(r.get("superseded_by", "")).strip() and not str(r.get("effective_to", "")).strip():
                errors.append(f"{tag}：{r.get('validity')} 但既无 effective_to 也无 superseded_by，无法定位失效时点")

        eff = str(r.get("effective_date", "") or "").strip()
        art_eff = str(r.get("article_effective_date", "") or "").strip()
        to = str(r.get("effective_to", "") or "").strip()
        if eff and to and DATE_RE.match(eff) and DATE_RE.match(to) and eff > to:
            errors.append(f"{tag}：生效日({eff})晚于失效日({to})")
        if art_eff and eff and DATE_RE.match(art_eff) and DATE_RE.match(eff) and art_eff > eff:
            errors.append(f"{tag}：条文首次施行日({art_eff})晚于当前修正版施行日({eff})")

        if r.get("partially_invalid") and not str(r.get("invalid_parts", "")).strip():
            errors.append(f"{tag}：标记部分失效但未指明 invalid_parts")

        # 未经核验的条文必须有说明，且不得进入自动对外通道
        if r.get("verified") is False and not str(r.get("note", "")).strip():
            errors.append(f"{tag}：verified=false 但缺少 note 说明")

        if h != "司法解释" and not str(r.get("scope", "")).strip():
            warns.append(f"{tag}：未标注 scope（特别/一般），若未列入成对关系表则无法按字段适用《立法法》第103条")

    # ——— 版本时间线：不允许重叠（重叠意味着同一时点有两个有效版本）———
    tl = build_version_timelines()
    for law, items in tl.items():
        span = []
        for v in items:
            f_, t_ = v.get("effective_from"), v.get("effective_to")
            if not f_:
                errors.append(f"《{law}》：版本缺少生效日")
                continue
            span.append((f_, t_ or "9999-12-31", v.get("version")))
        span.sort()
        for a, b in zip(span, span[1:]):
            if a[1] >= b[0]:
                errors.append(f"《{law}》：版本区间重叠 {a[2]}(~{a[1]}) 与 {b[2]}({b[0]}~)")

    # ——— 覆盖面统计 ———
    n = len(rows)
    hier = {}
    for r in rows:
        hier[r.get("hierarchy", "")] = hier.get(r.get("hierarchy", ""), 0) + 1
    scope_cov = sum(1 for r in rows if str(r.get("scope", "")).strip())
    art_eff_cov = sum(1 for r in rows if str(r.get("article_effective_date", "")).strip())
    para_cov = sum(1 for r in rows if str(r.get("paragraph", "")).strip())
    unverified = sum(1 for r in rows if r.get("verified") is False)

    print("=== 法规库体检报告 ===")
    print(f"条文总数：{n}   法律/规范数：{len(tl)}")
    print(f"位阶分布：{hier}")
    print(f"scope 覆盖：{scope_cov}/{n}   条文原始施行日覆盖：{art_eff_cov}/{n}   款项粒度：{para_cov}/{n}   待核验：{unverified}/{n}")

    if warns:
        print(f"\n告警 {len(warns)} 项（不阻断）：")
        for w in warns[:15]:
            print(f"  ⚠️  {w}")
        if len(warns) > 15:
            print(f"  … 其余 {len(warns) - 15} 项省略")

    if errors:
        print(f"\n错误 {len(errors)} 项：")
        for e in errors:
            print(f"  ❌ {e}")
        print("\n❌ 数据体检未通过：规则引擎将因元数据缺失或错误而失效")
        return 1

    print("\n✅ 数据体检通过：关键元数据完整、无重复条文、制定机关准确、版本时间线无重叠")
    return 0


if __name__ == "__main__":
    sys.exit(main())
