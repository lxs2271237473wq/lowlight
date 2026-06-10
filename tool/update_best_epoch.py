from pathlib import Path
import csv
import argparse


def read_results_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Ultralytics 的 results.csv 有些列名前面可能带空格，这里统一清理
            clean = {k.strip(): v for k, v in row.items()}
            rows.append(clean)

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    return rows


def to_float(x, default=None):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def pick(row, *keys):
    for k in keys:
        if k in row and row[k] not in ("", None):
            return row[k]
    return ""


def find_best_row(rows):
    # 优先用 mAP50-95 作为最优标准
    metric_candidates = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "map50_95",
        "mAP50-95",
    ]

    best_idx = 0
    best_score = -1.0
    best_metric_name = ""

    for i, row in enumerate(rows):
        score = None
        metric_name = ""

        for key in metric_candidates:
            if key in row:
                score = to_float(row.get(key), None)
                metric_name = key
                break

        # 如果没有 mAP50-95，就退回 mAP50
        if score is None:
            for key in ["metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50"]:
                if key in row:
                    score = to_float(row.get(key), None)
                    metric_name = key
                    break

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_idx = i
            best_metric_name = metric_name

    return best_idx, rows[best_idx], best_score, best_metric_name


def read_existing_csv(path):
    path = Path(path)
    if not path.exists():
        return [], []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    return rows, columns


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def upsert_summary(summary_path, row, key="experiment_name"):
    rows, old_columns = read_existing_csv(summary_path)

    new_columns = [
        "experiment_name",
        "stage",
        "module",
        "model",
        "dataset",
        "epochs",
        "imgsz",
        "batch",

        "best_epoch",
        "best_metric",
        "best_precision",
        "best_recall",
        "best_map50",
        "best_map50_95",

        "final_epoch",
        "final_precision",
        "final_recall",
        "final_map50",
        "final_map50_95",

        "result_dir",
        "best_weight",
        "note",
    ]

    # 保留旧表中可能已有但这里没列出的列
    columns = []
    for c in new_columns + old_columns:
        if c not in columns:
            columns.append(c)

    updated = False
    output_rows = []

    for old in rows:
        if old.get(key) == row.get(key):
            merged = dict(old)
            merged.update(row)
            output_rows.append(merged)
            updated = True
        else:
            output_rows.append(old)

    if not updated:
        output_rows.append(row)

    write_csv(summary_path, output_rows, columns)


def build_summary_row(
    experiment_name,
    stage,
    module,
    model,
    dataset,
    epochs,
    imgsz,
    batch,
    result_dir,
    note,
):
    result_dir = Path(result_dir)
    results_csv = result_dir / "results.csv"
    best_weight = result_dir / "weights" / "best.pt"

    rows = read_results_csv(results_csv)
    best_idx, best, best_score, best_metric_name = find_best_row(rows)
    final = rows[-1]

    best_epoch = pick(best, "epoch")
    final_epoch = pick(final, "epoch")

    # 有些 Ultralytics 的 epoch 从 0 开始，这里保留原始值，不强行 +1。
    # 论文或表格里如果想写第几轮，可以后续统一处理。
    row = {
        "experiment_name": experiment_name,
        "stage": stage,
        "module": module,
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,

        "best_epoch": best_epoch,
        "best_metric": best_metric_name,
        "best_precision": pick(best, "metrics/precision(B)", "metrics/precision", "precision"),
        "best_recall": pick(best, "metrics/recall(B)", "metrics/recall", "recall"),
        "best_map50": pick(best, "metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50"),
        "best_map50_95": pick(best, "metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "mAP50-95"),

        "final_epoch": final_epoch,
        "final_precision": pick(final, "metrics/precision(B)", "metrics/precision", "precision"),
        "final_recall": pick(final, "metrics/recall(B)", "metrics/recall", "recall"),
        "final_map50": pick(final, "metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50"),
        "final_map50_95": pick(final, "metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "mAP50-95"),

        "result_dir": str(result_dir),
        "best_weight": str(best_weight) if best_weight.exists() else "",
        "note": note,
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--imgsz", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--module_summary", required=True)
    parser.add_argument("--root_summary", required=True)

    args = parser.parse_args()

    row = build_summary_row(
        experiment_name=args.experiment_name,
        stage=args.stage,
        module=args.module,
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        result_dir=args.result_dir,
        note=args.note,
    )

    upsert_summary(args.module_summary, row)
    upsert_summary(args.root_summary, row)

    print("Updated summary tables:")
    print(args.module_summary)
    print(args.root_summary)
    print("Best epoch:", row["best_epoch"])
    print("Best mAP50:", row["best_map50"])
    print("Best mAP50-95:", row["best_map50_95"])


if __name__ == "__main__":
    main()
