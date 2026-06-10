from pathlib import Path
import yaml
from ultralytics import YOLO

from tool.update_best_epoch import build_summary_row, upsert_summary


def load_yaml(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        raise ValueError(f"experiment.name 不能包含路径符号: {exp_name}")

    if "/" in module or "\\" in module:
        raise ValueError(f"experiment.module 不能包含路径符号: {module}")

    project_root = root / out.get("project_root", "runs")
    project_dir = project_root / module
    result_dir = project_dir / exp_name

    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = root / tr["model"] if not Path(tr["model"]).is_absolute() else Path(tr["model"])
    data_path = root / tr["data"] if not Path(tr["data"]).is_absolute() else Path(tr["data"])

    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"数据集 yaml 不存在: {data_path}")

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Stage: {stage}")
    print(f"Module: {module}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Epochs: {tr.get('epochs')}")
    print(f"Image size: {tr.get('imgsz')}")
    print(f"Batch: {tr.get('batch')}")
    print(f"Project dir: {project_dir}")
    print(f"Experiment name: {exp_name}")
    print(f"Expected result dir: {result_dir}")
    print("=" * 80)

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=tr.get("epochs", 100),
        imgsz=tr.get("imgsz", 640),
        batch=tr.get("batch", 16),
        device=tr.get("device", 0),
        workers=tr.get("workers", 8),
        seed=tr.get("seed", 42),
        optimizer=tr.get("optimizer", "auto"),
        lr0=tr.get("lr0", 0.01),
        patience=tr.get("patience", 50),
        pretrained=tr.get("pretrained", True),
        save_period=out.get("save_period", -1),
        project=str(project_dir),
        name=exp_name,
        exist_ok=tr.get("exist_ok", True),
    )

    if not result_dir.exists():
        raise RuntimeError(
            f"训练结束后没有找到预期结果目录: {result_dir}\n"
            f"请检查 Ultralytics 实际保存路径。"
        )

    module_summary = project_dir / f"{module}_summary.csv"
    root_summary = root / "baseline_summary.csv"

    row = build_summary_row(
        experiment_name=exp_name,
        stage=stage,
        module=module,
        model=str(Path(tr["model"])),
        dataset=str(Path(tr["data"])),
        epochs=str(tr.get("epochs", "")),
        imgsz=str(tr.get("imgsz", "")),
        batch=str(tr.get("batch", "")),
        result_dir=str(result_dir.relative_to(root)),
        note=note,
    )

    upsert_summary(module_summary, row)
    upsert_summary(root_summary, row)

    print("\nTraining finished.")
    print(f"Result directory: {result_dir}")
    print(f"Module summary: {module_summary}")
    print(f"Root summary: {root_summary}")
    print(f"Best epoch: {row.get('best_epoch')}")
    print(f"Best mAP50: {row.get('best_map50')}")
    print(f"Best mAP50-95: {row.get('best_map50_95')}")


if __name__ == "__main__":
    main()
