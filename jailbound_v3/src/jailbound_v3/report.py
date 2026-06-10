from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jailbound.loose_eval import refusal_reason


METRICS = [
    ("qwen3guard", "Qwen3Guard Unsafe ASR"),
    ("non_refusal", "Non-Refusal ASR"),
    ("qwen25_judge", "Qwen2.5 Judge ASR"),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _preferred_rows(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    for name in ("follow_asr_eval.jsonl", "loose_asr_eval.jsonl", "guard_eval.jsonl", "attack_results.jsonl"):
        path = run_dir / name
        rows = _read_jsonl(path)
        if rows:
            return name, rows
    raise FileNotFoundError(f"No evaluation JSONL found under {run_dir}")


def _metric_value(row: dict[str, Any], metric: str) -> bool | None:
    if metric == "qwen3guard":
        if "asr_success" in row:
            return bool(row["asr_success"])
        label = row.get("guard_label")
        if label is not None:
            return str(label).lower() == "unsafe"
        return None
    if metric == "non_refusal":
        if "non_refusal_success" in row:
            return bool(row["non_refusal_success"])
        if "response" in row:
            return refusal_reason(row.get("response", "")) is None
        return None
    if metric == "qwen25_judge":
        value = row.get("follow_judge_success")
        if value is None:
            return None
        return bool(value)
    raise KeyError(metric)


def _rate(success: int, total: int) -> float | None:
    return success / total if total else None


def _fmt_rate(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _summarize_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"run": label, "source_rows": len(rows)}
    for metric, _title in METRICS:
        counted = [x for x in (_metric_value(row, metric) for row in rows) if x is not None]
        success = sum(1 for x in counted if x)
        total = len(counted)
        summary[f"{metric}_success"] = success
        summary[f"{metric}_total"] = total
        summary[f"{metric}_asr"] = _rate(success, total)
    return summary


def _summarize_categories(label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "unknown"))].append(row)

    out = []
    for category, items in sorted(grouped.items()):
        record: dict[str, Any] = {"run": label, "category": category, "source_rows": len(items)}
        for metric, _title in METRICS:
            counted = [x for x in (_metric_value(row, metric) for row in items) if x is not None]
            success = sum(1 for x in counted if x)
            total = len(counted)
            record[f"{metric}_success"] = success
            record[f"{metric}_total"] = total
            record[f"{metric}_asr"] = _rate(success, total)
        out.append(record)
    return out


def _comparison_rows(overall: list[dict[str, Any]], category_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(overall) < 2:
        return [], []
    base = overall[0]
    comparisons = []
    for current in overall[1:]:
        row = {"baseline": base["run"], "run": current["run"]}
        for metric, _title in METRICS:
            base_value = base.get(f"{metric}_asr")
            current_value = current.get(f"{metric}_asr")
            row[f"{metric}_baseline_asr"] = base_value
            row[f"{metric}_run_asr"] = current_value
            row[f"{metric}_delta"] = (
                current_value - base_value if current_value is not None and base_value is not None else None
            )
        comparisons.append(row)

    by_run_category = {(row["run"], row["category"]): row for row in category_rows}
    categories = sorted({row["category"] for row in category_rows})
    category_comparisons = []
    for current in overall[1:]:
        for category in categories:
            base_row = by_run_category.get((base["run"], category))
            current_row = by_run_category.get((current["run"], category))
            if not base_row or not current_row:
                continue
            row = {"baseline": base["run"], "run": current["run"], "category": category}
            for metric, _title in METRICS:
                base_value = base_row.get(f"{metric}_asr")
                current_value = current_row.get(f"{metric}_asr")
                row[f"{metric}_baseline_asr"] = base_value
                row[f"{metric}_run_asr"] = current_value
                row[f"{metric}_delta"] = (
                    current_value - base_value if current_value is not None and base_value is not None else None
                )
            category_comparisons.append(row)
    return comparisons, category_comparisons


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _overall_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# JailBound ASR Summary",
        "",
        "| Run | Rows | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {rows} | {guard} | {non_refusal} | {judge} |".format(
                run=row["run"],
                rows=row["source_rows"],
                guard=_fmt_rate(row.get("qwen3guard_asr")),
                non_refusal=_fmt_rate(row.get("non_refusal_asr")),
                judge=_fmt_rate(row.get("qwen25_judge_asr")),
            )
        )
    return "\n".join(lines) + "\n"


def _category_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# JailBound Category ASR",
        "",
        "| Run | Category | Rows | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {category} | {rows} | {guard} | {non_refusal} | {judge} |".format(
                run=row["run"],
                category=row["category"],
                rows=row["source_rows"],
                guard=_fmt_rate(row.get("qwen3guard_asr")),
                non_refusal=_fmt_rate(row.get("non_refusal_asr")),
                judge=_fmt_rate(row.get("qwen25_judge_asr")),
            )
        )
    return "\n".join(lines) + "\n"


def _comparison_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# JailBound Before/After Comparison",
        "",
        "| Baseline | Run | Qwen3Guard Delta | Non-Refusal Delta | Qwen2.5 Judge Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {baseline} | {run} | {guard} | {non_refusal} | {judge} |".format(
                baseline=row["baseline"],
                run=row["run"],
                guard=_fmt_rate(row.get("qwen3guard_delta")),
                non_refusal=_fmt_rate(row.get("non_refusal_delta")),
                judge=_fmt_rate(row.get("qwen25_judge_delta")),
            )
        )
    return "\n".join(lines) + "\n"


def build_report(runs: list[tuple[str, Path]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    source_map = {}
    for label, run_dir in runs:
        source, rows = _preferred_rows(run_dir)
        loaded.append((label, rows))
        source_map[label] = {"dir": str(run_dir), "source": source, "rows": len(rows)}

    overall = [_summarize_rows(label, rows) for label, rows in loaded]
    category_rows = []
    for label, rows in loaded:
        category_rows.extend(_summarize_categories(label, rows))
    comparisons, category_comparisons = _comparison_rows(overall, category_rows)

    paths = {
        "overall_csv": output_dir / "overall_asr.csv",
        "category_csv": output_dir / "category_asr.csv",
        "comparison_csv": output_dir / "before_after_asr.csv",
        "category_comparison_csv": output_dir / "before_after_category_asr.csv",
        "overall_md": output_dir / "overall_asr.md",
        "category_md": output_dir / "category_asr.md",
        "comparison_md": output_dir / "before_after_asr.md",
        "summary_json": output_dir / "report_summary.json",
    }

    _write_csv(paths["overall_csv"], overall)
    _write_csv(paths["category_csv"], category_rows)
    _write_csv(paths["comparison_csv"], comparisons)
    _write_csv(paths["category_comparison_csv"], category_comparisons)
    paths["overall_md"].write_text(_overall_md(overall), encoding="utf-8")
    paths["category_md"].write_text(_category_md(category_rows), encoding="utf-8")
    paths["comparison_md"].write_text(_comparison_md(comparisons), encoding="utf-8")
    paths["summary_json"].write_text(
        json.dumps(
            {
                "runs": source_map,
                "overall": overall,
                "category": category_rows,
                "comparison": comparisons,
                "category_comparison": category_comparisons,
                "metric_note": {
                    "qwen3guard": "Strict Unsafe rate from Qwen3Guard.",
                    "non_refusal": "Paper-style non-refusal rate, computed by refusal template matching.",
                    "qwen25_judge": "Instruction-following ASR from follow_asr_eval.jsonl.",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths



