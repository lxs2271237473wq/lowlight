from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import csv

SUMMARY_COLUMNS: List[str] = [
    "experiment_name",
    "stage",
    "module",
    "model",
    "dataset",
    "epochs",
    "imgsz",
    "batch",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "params",
    "gflops",
    "fps",
    "weight_path",
    "result_dir",
    "is_best",
    "note",
]


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {col: row.get(col, "") for col in SUMMARY_COLUMNS}


def ensure_summary_table(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
    return path


def append_summary_row(path: str | Path, row: Dict[str, Any]) -> None:
    path = ensure_summary_table(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writerow(_normalise_row(row))


def read_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalise_row(row))


def update_global_best(global_table: str | Path, row: Dict[str, Any]) -> None:
    """Keep one best row per module in the global table.

    Baselines are retained as baseline records. For non-baseline modules, this
    function replaces the module row only when the new mAP50-95 is higher.
    """
    global_table = ensure_summary_table(global_table)
    rows = read_rows(global_table)
    module = str(row.get("module", ""))
    stage = str(row.get("stage", ""))

    if stage == "baseline" or module == "baselines":
        rows.append(_normalise_row(row))
        write_rows(global_table, rows)
        return

    new_score = _to_float(row.get("map50_95"))
    replaced = False
    out_rows: List[Dict[str, Any]] = []
    for old in rows:
        if old.get("module") != module:
            out_rows.append(old)
            continue
        old_score = _to_float(old.get("map50_95"))
        if new_score >= old_score:
            out_rows.append(_normalise_row(row))
        else:
            out_rows.append(old)
        replaced = True

    if not replaced:
        out_rows.append(_normalise_row(row))

    write_rows(global_table, out_rows)


def _to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("-inf")
        return float(value)
    except Exception:
        return float("-inf")
