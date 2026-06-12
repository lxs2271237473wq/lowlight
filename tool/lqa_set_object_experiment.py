#模块一想法二模块消融实验，主要是对object_level和batch_level_weight两个参数进行消融，看看它们对性能的影响。
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

CONFIGS = {
    "object_a025": {
        "name": "exdark_yolo11s_lqa_loss_object_a025",
        "alpha": 0.25,
        "max_weight": 1.25,
        "warmup_epochs": 20,
        "small_area_thr": 0.01,
        "darkness_weight": 0.60,
        "small_weight": 0.40,
        "object_level": True,
        "batch_level_weight": False,
    },
    "object_a020": {
        "name": "exdark_yolo11s_lqa_loss_object_a020",
        "alpha": 0.20,
        "max_weight": 1.20,
        "warmup_epochs": 20,
        "small_area_thr": 0.01,
        "darkness_weight": 0.60,
        "small_weight": 0.40,
        "object_level": True,
        "batch_level_weight": False,
    },
}


def set_key_recursive(obj: Any, key: str, value: Any) -> bool:
    changed = False
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == key:
                obj[k] = value
                changed = True
            else:
                changed = set_key_recursive(obj[k], key, value) or changed
    elif isinstance(obj, list):
        for item in obj:
            changed = set_key_recursive(item, key, value) or changed
    return changed


def ensure_dict(cfg: dict, key: str) -> dict:
    if key not in cfg or not isinstance(cfg[key], dict):
        cfg[key] = {}
    return cfg[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--file", default="train_config.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    path = Path(args.file)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must contain a YAML mapping")

    exp = CONFIGS[args.config_name]
    exp_name = exp["name"]

    for k in ["name", "exp_name", "experiment_name"]:
        set_key_recursive(cfg, k, exp_name)
    cfg["name"] = exp_name
    cfg["exp_name"] = exp_name
    cfg["experiment_name"] = exp_name

    for k, v in {
        "epochs": args.epochs,
        "workers": args.workers,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
    }.items():
        if not set_key_recursive(cfg, k, v):
            cfg[k] = v

    for section_name in ["lqa_loss", "lqa"]:
        section = ensure_dict(cfg, section_name)
        section["enabled"] = True
        for k in [
            "alpha",
            "max_weight",
            "warmup_epochs",
            "small_area_thr",
            "darkness_weight",
            "small_weight",
            "object_level",
            "batch_level_weight",
        ]:
            section[k] = exp[k]

    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"set config: {args.config_name}")
    print(f"experiment: {exp_name}")
    print(
        "params:",
        f"alpha={exp['alpha']}",
        f"max_weight={exp['max_weight']}",
        f"darkness_weight={exp['darkness_weight']}",
        f"small_weight={exp['small_weight']}",
        f"warmup_epochs={exp['warmup_epochs']}",
        f"object_level={exp['object_level']}",
        f"batch_level_weight={exp['batch_level_weight']}",
    )
    print(
        "formal:",
        f"epochs={args.epochs}",
        f"workers={args.workers}",
        f"patience={args.patience}",
        f"batch={args.batch}",
        f"imgsz={args.imgsz}",
    )


if __name__ == "__main__":
    main()
