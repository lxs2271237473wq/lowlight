from pathlib import Path
import csv
import yaml
from ultralytics import YOLO

from tool.update_best_epoch import build_summary_row, upsert_summary


def load_yaml(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_float(value, default=-1.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return [], []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv_rows(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def upsert_root_summary(root_summary, row):
    """Root summary rule:
    - baselines: keep all baseline rows.
    - screening: never enter root summary.
    - modules: keep only the best row for each module by best_map50_95.
    """
    root_summary = Path(root_summary)
    stage = row.get("stage", "")
    module = row.get("module", "")
    exp_name = row.get("experiment_name", "")

    if stage == "screening":
        print("Skip root summary update because stage == screening.")
        return

    if stage == "baseline" or module == "baselines":
        upsert_summary(root_summary, row)
        return

    rows, old_columns = read_csv_rows(root_summary)

    new_columns = list(row.keys())
    columns = []
    for c in old_columns + new_columns:
        if c not in columns:
            columns.append(c)

    existing_module_rows = [r for r in rows if r.get("module") == module]
    other_rows = [r for r in rows if r.get("module") != module]

    new_score = to_float(row.get("best_map50_95"))

    if existing_module_rows:
        best_existing = max(
            existing_module_rows,
            key=lambda r: to_float(r.get("best_map50_95"))
        )
        old_score = to_float(best_existing.get("best_map50_95"))

        if old_score > new_score:
            print(
                f"Skip root summary update for module={module}. "
                f"Existing best mAP50-95={old_score} is higher than new={new_score}."
            )
            return

    other_rows.append(row)
    write_csv_rows(root_summary, other_rows, columns)
    print(f"Root summary updated with best row for module={module}: {exp_name}")


def main():
    root = Path.cwd().resolve()
    cfg = load_yaml(root / "train_config.yaml")

    exp = cfg["experiment"]
    tr = cfg["train"]
    out = cfg["output"]

    exp_name = exp["name"]
    stage = exp.get("stage", "")
    module = exp["module"]
    note = exp.get("note", "")

    if "/" in exp_name or "\\" in exp_name:
        raise ValueError(f"experiment.name cannot contain path separators: {exp_name}")

    if "/" in module or "\\" in module:
        raise ValueError(f"experiment.module cannot contain path separators: {module}")

    project_root = root / out.get("project_root", "runs")
    project_dir = project_root / module
    result_dir = project_dir / exp_name

    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(tr["model"])
    data_path = Path(tr["data"])

    if not model_path.is_absolute():
        model_path = root / model_path

    if not data_path.is_absolute():
        data_path = root / data_path

    if not model_path.exists():
        raise FileNotFoundError(f"model does not exist: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"data yaml does not exist: {data_path}")

    epochs = int(tr.get("epochs", 100))

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Stage: {stage}")
    print(f"Module: {module}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {tr.get('imgsz')}")
    print(f"Batch: {tr.get('batch')}")
    print(f"Project dir: {project_dir}")
    print(f"Expected result dir: {result_dir}")
    print("=" * 80)

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=tr.get("imgsz", 640),
        batch=tr.get("batch", 16),
        device=tr.get("device", 0),
        workers=tr.get("workers", 8),
        seed=tr.get("seed", 42),
        optimizer=tr.get("optimizer", "auto"),
        lr0=tr.get("lr0", 0.01),
        patience=tr.get("patience", 100),
        pretrained=tr.get("pretrained", True),
        save_period=out.get("save_period", -1),
        project=str(project_dir),
        name=exp_name,
        exist_ok=tr.get("exist_ok", True),
    )

    if not result_dir.exists():
        raise RuntimeError(f"Expected result directory was not found: {result_dir}")

    if epochs < 100:
        print("\nDebug/path-test run finished.")
        print(f"Result directory: {result_dir}")
        print(f"Epochs: {epochs}")
        print("Skip summary update because epochs < 100.")
        return

    module_summary = project_dir / f"{module}_summary.csv"
    root_summary = root / "baseline_summary.csv"

    row = build_summary_row(
        experiment_name=exp_name,
        stage=stage,
        module=module,
        model=str(Path(tr["model"])),
        dataset=str(Path(tr["data"])),
        epochs=str(epochs),
        imgsz=str(tr.get("imgsz", "")),
        batch=str(tr.get("batch", "")),
        result_dir=str(result_dir.relative_to(root)),
        note=note,
    )

    upsert_summary(module_summary, row)

    if stage == "screening":
        print("Screening run: module summary updated, root summary skipped.")
    else:
        upsert_root_summary(root_summary, row)

    print("\nTraining finished.")
    print(f"Result directory: {result_dir}")
    print(f"Module summary: {module_summary}")
    print(f"Root summary: {root_summary}")
    print(f"Best epoch: {row.get('best_epoch')}")
    print(f"Best mAP50: {row.get('best_map50')}")
    print(f"Best mAP50-95: {row.get('best_map50_95')}")


if __name__ == "__main__":
    main()
