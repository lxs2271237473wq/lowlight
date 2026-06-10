from __future__ import annotations

import argparse
from pathlib import Path
from tool.table_utils import append_summary_row, update_global_best, ensure_summary_table, SUMMARY_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Append an experiment result to module and global summary tables.")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--epochs", default="")
    parser.add_argument("--imgsz", default="")
    parser.add_argument("--batch", default="")
    parser.add_argument("--precision", default="")
    parser.add_argument("--recall", default="")
    parser.add_argument("--map50", default="")
    parser.add_argument("--map50_95", default="")
    parser.add_argument("--params", default="")
    parser.add_argument("--gflops", default="")
    parser.add_argument("--fps", default="")
    parser.add_argument("--weight_path", default="")
    parser.add_argument("--result_dir", default="")
    parser.add_argument("--is_best", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    row = {k: getattr(args, k) for k in SUMMARY_COLUMNS}

    module_dir = Path("runs") / args.module
    module_table = module_dir / f"{args.module}_summary.csv"
    ensure_summary_table(module_table)
    append_summary_row(module_table, row)
    update_global_best("baseline_summary.csv", row)

    print(f"Updated module table: {module_table}")
    print("Updated global table: baseline_summary.csv")


if __name__ == "__main__":
    main()
