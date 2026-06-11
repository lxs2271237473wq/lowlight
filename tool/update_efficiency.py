from pathlib import Path
import argparse
import csv
import pandas as pd
from ultralytics import YOLO


def update_table(table_path, experiment_name, metrics):
    table_path = Path(table_path)

    if not table_path.exists():
        print(f"skip missing table: {table_path}")
        return

    df = pd.read_csv(table_path)

    if "experiment_name" not in df.columns:
        print(f"skip invalid table: {table_path}")
        return

    for k, v in metrics.items():
        if k not in df.columns:
            df[k] = ""

    mask = df["experiment_name"] == experiment_name

    if not mask.any():
        print(f"experiment not found in {table_path}: {experiment_name}")
        return

    for k, v in metrics.items():
        df.loc[mask, k] = v

    df.to_csv(table_path, index=False)
    print(f"updated: {table_path}")


def get_file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return ""
    return round(path.stat().st_size / 1024 / 1024, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--module_summary", required=True)
    parser.add_argument("--root_summary", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(str(model_path))

    print("=" * 80)
    print(f"Efficiency test: {args.experiment_name}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print("=" * 80)

    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        save_json=False,
        verbose=False,
    )

    speed = getattr(results, "speed", {}) or {}

    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))

    latency_ms = preprocess_ms + inference_ms + postprocess_ms

    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    params_m = ""
    gflops = ""

    try:
        info = model.info(verbose=False)
        # Ultralytics 不同版本返回格式可能不同，这里只做保守兼容
        if isinstance(info, tuple):
            # common format may include layers, params, gradients, flops
            if len(info) >= 2:
                params_m = round(float(info[1]) / 1e6, 4)
            if len(info) >= 4:
                gflops = round(float(info[3]), 4)
    except Exception:
        pass

    metrics = {
        "params_m": params_m,
        "gflops": gflops,
        "preprocess_ms": round(preprocess_ms, 4),
        "inference_ms": round(inference_ms, 4),
        "postprocess_ms": round(postprocess_ms, 4),
        "latency_ms": round(latency_ms, 4),
        "fps": round(fps, 4),
        "inference_fps": round(inference_fps, 4),
        "model_size_mb": get_file_size_mb(model_path),
        "efficiency_imgsz": args.imgsz,
        "efficiency_batch": args.batch,
        "efficiency_device": args.device,
    }

    print("Efficiency metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    update_table(args.module_summary, args.experiment_name, metrics)
    update_table(args.root_summary, args.experiment_name, metrics)


if __name__ == "__main__":
    main()
