from __future__ import annotations

import argparse
from pathlib import Path
from tool.table_utils import ensure_summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new module directory only when needed.")
    parser.add_argument("module", help="Example: module1_lqa_loss")
    args = parser.parse_args()

    module = args.module.strip().strip("/")
    if not module:
        raise SystemExit("Module name cannot be empty.")

    root = Path("runs") / module
    root.mkdir(parents=True, exist_ok=True)
    table = root / f"{module}_summary.csv"
    ensure_summary_table(table)

    print(f"Created module directory: {root}")
    print(f"Created module summary table: {table}")


if __name__ == "__main__":
    main()
