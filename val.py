from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

from tool.config_utils import load_yaml, ensure_dir
from tool.result_utils import extract_train_metrics


def main() -> None:
    cfg = load_yaml("val_config.yaml")
    exp = cfg.get("experiment", {})
    val = cfg.get("val", {})
    output = cfg.get("output", {})

    name = exp.get("name", "val_result")
    module = exp.get("module", "baselines")
    project_root = Path(output.get("project_root", "runs"))
    project = ensure_dir(project_root / module)

    model_path = val.get("model")
    if not model_path:
        raise ValueError("val.model is required in val_config.yaml")

    model = YOLO(model_path)

    val_args = dict(val)
    val_args.pop("model", None)
    val_args["project"] = str(project)
    val_args["name"] = name

    result = model.val(**val_args)
    metrics = extract_train_metrics(result)

    print(f"\nValidation finished: {name}")
    print(f"Result directory: {project / name}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
