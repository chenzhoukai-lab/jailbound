from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jailbound.loose_eval import FOLLOW_JUDGE_PROMPT  # noqa: E402


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def count_rate(count: int | None, total: int | None, rate: float | None) -> str:
    if count is None or total is None:
        return pct(rate)
    return f"{count}/{total} ({pct(rate)})"


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def overall_rows(summary: dict[str, Any]) -> list[list[str]]:
    total = int(summary.get("total", 0))
    return [
        [
            "Non-refusal ASR",
            count_rate(summary.get("non_refusal_success"), total, summary.get("non_refusal_asr")),
            "Response is non-empty and does not match refusal templates.",
        ],
        [
            "Heuristic follow ASR",
            count_rate(summary.get("follow_heuristic_success"), total, summary.get("follow_heuristic_asr")),
            "Rule-based middle metric; filters short/generic/safe-description responses.",
        ],
        [
            "Qwen2.5 follow judge ASR",
            count_rate(summary.get("follow_judge_success"), total, summary.get("follow_judge_asr")),
            "Qwen2.5-7B-Instruct judges whether the response semantically follows the original harmful instruction.",
        ],
        [
            "Qwen3Guard Unsafe ASR",
            count_rate(summary.get("qwen3guard_unsafe"), total, summary.get("qwen3guard_asr")),
            "Strict safety metric: Qwen3Guard labels the response Unsafe.",
        ],
    ]


def category_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    details = summary.get("category_detail", {})
    for category, item in sorted(details.items()):
        total = int(item.get("total", 0))
        rows.append(
            [
                category,
                str(total),
                count_rate(item.get("non_refusal_success"), total, item.get("non_refusal_asr")),
                count_rate(item.get("follow_heuristic_success"), total, item.get("follow_heuristic_asr")),
                count_rate(item.get("follow_judge_success"), total, item.get("follow_judge_asr")),
                count_rate(item.get("qwen3guard_unsafe"), total, item.get("qwen3guard_asr")),
            ]
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", "<br>") for cell in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall_headers = ["Metric", "Value", "Meaning"]
    category_headers = [
        "Category",
        "Total",
        "Non-refusal ASR",
        "Heuristic follow ASR",
        "Qwen2.5 follow judge ASR",
        "Qwen3Guard Unsafe ASR",
    ]
    text = f"""# JailBound ASR Summary

## Overall

{markdown_table(overall_headers, overall_rows(summary))}

## By Category

{markdown_table(category_headers, category_rows(summary))}

## Qwen2.5 Follow Judge Prompt

This prompt produced the semantic-following ASR reported as `follow_judge_asr`
in `follow_asr_summary.json` (for the current run: `{pct(summary.get("follow_judge_asr"))}`).

```text
{FOLLOW_JUDGE_PROMPT}
```

## Notes

- `Non-refusal ASR` is the loosest metric and only checks whether the model avoids explicit refusal templates.
- `Qwen2.5 follow judge ASR` is the middle metric: it asks a general instruction model whether the response semantically follows the original harmful instruction.
- `Qwen3Guard Unsafe ASR` is the strict metric: it only counts responses Qwen3Guard labels as `Unsafe`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JailBound ASR summaries as Markdown and CSV tables.")
    parser.add_argument(
        "--summary",
        default="outputs/qwen25vl_jailbound/follow_asr_summary.json",
        help="Path to follow_asr_summary.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/qwen25vl_jailbound/tables",
        help="Directory for exported tables.",
    )
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    out_dir = Path(args.out_dir)
    md_path = out_dir / "asr_summary.md"
    overall_csv = out_dir / "asr_overall.csv"
    category_csv = out_dir / "asr_by_category.csv"

    write_markdown(md_path, summary)
    write_csv(overall_csv, ["Metric", "Value", "Meaning"], overall_rows(summary))
    write_csv(
        category_csv,
        ["Category", "Total", "Non-refusal ASR", "Heuristic follow ASR", "Qwen2.5 follow judge ASR", "Qwen3Guard Unsafe ASR"],
        category_rows(summary),
    )

    print(f"Saved Markdown report: {md_path}")
    print(f"Saved overall CSV: {overall_csv}")
    print(f"Saved category CSV: {category_csv}")


if __name__ == "__main__":
    main()

