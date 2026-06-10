from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

from tool.config_utils import load_yaml, ensure_dir
from tool.result_utils import extract_train_metrics, find_best_weight
from tool.table_utils import ensure_summary_table, append_summary_row, update_global_best


def main() -> None:
    cfg = load_yaml("train_config.yaml")
    exp = cfg.get("experiment", {})
    train = cfg.get("train", {})
    output = cfg.get("output", {})

    name = exp.get("name", "unnamed_experiment")
    stage = exp.get("stage", "baseline")
    module = exp.get("module", "baselines")
    note = exp.get("note", "")

    project_root = Path(output.get("project_root", "runs"))
    project = ensure_dir(project_root / module)
    module_table = project / f"{module}_summary.csv"
    ensure_summary_table(module_table)
    ensure_summary_table("baseline_summary.csv")

    model_path = train.get("model")
    if not model_path:
        raise ValueError("train.model is required in train_config.yaml")

    data_yaml = train.get("data")
    if not data_yaml:
        raise ValueError("train.data is required in train_config.yaml")

    model = YOLO(model_path)

    train_args = dict(train)
    train_args.pop("model", None)
    train_args["project"] = str(project)
    train_args["name"] = name
    train_args["save_period"] = output.get("save_period", -1)

    result = model.train(**train_args)

    result_dir = project / name
    metrics = extract_train_metrics(result)
    weight_path = find_best_weight(result_dir)

    row = {
        "experiment_name": name,
        "stage": stage,
        "module": module,
        "model": str(model_path),
        "dataset": str(data_yaml),
        "epochs": train.get("epochs", ""),
        "imgsz": train.get("imgsz", ""),
        "batch": train.get("batch", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "map50": metrics.get("map50", ""),
        "map50_95": metrics.get("map50_95", ""),
        "params": metrics.get("params", ""),
        "gflops": metrics.get("gflops", ""),
        "fps": metrics.get("fps", ""),
        "weight_path": weight_path,
        "result_dir": str(result_dir),
        "is_best": "",
        "note": note,
    }

    append_summary_row(module_table, row)
    update_global_best("baseline_summary.csv", row)

    print(f"\nTraining finished: {name}")
    print(f"Result directory: {result_dir}")
    print(f"Module table: {module_table}")
    print("Global table: baseline_summary.csv")


if __name__ == "__main__":
    main()
