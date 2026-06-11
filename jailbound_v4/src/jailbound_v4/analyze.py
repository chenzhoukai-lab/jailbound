from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_by_category(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("category", "unknown")].append(row)
    summary = []
    for category, items in sorted(buckets.items()):
        total = len(items)
        asr = sum(1 for x in items if x.get("asr_success"))
        effective = sum(1 for x in items if x.get("attack_effective"))
        summary.append(
            {
                "category": category,
                "total": total,
                "asr_success": asr,
                "asr": asr / total if total else 0.0,
                "attack_effective": effective,
                "attack_effective_rate": effective / total if total else 0.0,
            }
        )
    return summary


def write_category_csv(summary: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["category", "total"])
        writer.writeheader()
        writer.writerows(summary)
    return out_path


def write_markdown_report(summary: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# JailBound v4 Category Analysis",
        "",
        "| Category | Total | Guard ASR | Attack Effective |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['category']} | {row['total']} | {row['asr']:.2%} | {row['attack_effective_rate']:.2%} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path




