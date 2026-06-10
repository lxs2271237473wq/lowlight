from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def extract_train_metrics(result: Any) -> Dict[str, Any]:
    """Extract common YOLO metrics from Ultralytics train/val result objects.

    Ultralytics result attributes vary across versions, so this function uses
    conservative fallbacks. Missing values are kept blank rather than guessed.
    """
    metrics: Dict[str, Any] = {
        "precision": "",
        "recall": "",
        "map50": "",
        "map50_95": "",
        "params": "",
        "gflops": "",
        "fps": "",
    }

    box = getattr(result, "box", None)
    if box is not None:
        metrics["precision"] = _safe_attr(box, "mp")
        metrics["recall"] = _safe_attr(box, "mr")
        metrics["map50"] = _safe_attr(box, "map50")
        metrics["map50_95"] = _safe_attr(box, "map")

    speed = getattr(result, "speed", None)
    if isinstance(speed, dict):
        inference_ms = speed.get("inference")
        try:
            if inference_ms and float(inference_ms) > 0:
                metrics["fps"] = 1000.0 / float(inference_ms)
        except Exception:
            pass

    return metrics


def find_best_weight(result_dir: str | Path) -> str:
    result_dir = Path(result_dir)
    best = result_dir / "weights" / "best.pt"
    last = result_dir / "weights" / "last.pt"
    if best.exists():
        return str(best)
    if last.exists():
        return str(last)
    return ""


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return ""
