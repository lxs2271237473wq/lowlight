#!/usr/bin/env python3
"""Create an enhanced-val YOLO dataset for low-light diagnosis.

This script does NOT train a model. It creates a new YOLO dataset folder whose
validation images are enhanced copies of the original validation images, while
labels/classes are copied from the source dataset.

Typical use:
  python tool/make_enhanced_yolo_val.py \
    --src datasets_yolo/AODRaw_sRGB_YOLO \
    --dst datasets_yolo/AODRaw_sRGB_YOLO_enhanced_val \
    --method clahe_gamma \
    --gamma 0.75 \
    --clip-limit 2.0
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def enhance_clahe_gamma(img_bgr: np.ndarray, gamma: float = 0.75, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply LAB-CLAHE on luminance, then gamma brightening."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    gamma = float(gamma)
    gamma = max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(out, table)
    return out


def enhance_gamma(img_bgr: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def enhance_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def load_yaml(src: Path) -> Dict:
    yaml_candidates = sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"No yaml file found in {src}")
    with yaml_candidates[0].open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_source_yaml"] = str(yaml_candidates[0])
    return data


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source YOLO dataset root, e.g. datasets_yolo/AODRaw_sRGB_YOLO")
    parser.add_argument("--dst", required=True, help="Destination dataset root for enhanced val")
    parser.add_argument("--method", default="clahe_gamma", choices=["clahe_gamma", "clahe", "gamma"])
    parser.add_argument("--gamma", type=float, default=0.75, help="Gamma < 1 brightens dark images")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src}")

    src_val_img = src / "images" / "val"
    src_val_lbl = src / "labels" / "val"
    if not src_val_img.exists():
        raise FileNotFoundError(f"Missing source val images: {src_val_img}")
    if not src_val_lbl.exists():
        raise FileNotFoundError(f"Missing source val labels: {src_val_lbl}")

    if dst.exists() and args.overwrite:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    dst_img_val = dst / "images" / "val"
    dst_lbl_val = dst / "labels" / "val"
    dst_img_val.mkdir(parents=True, exist_ok=True)
    dst_lbl_val.mkdir(parents=True, exist_ok=True)

    # Copy labels exactly. They are not enhanced and must remain aligned by filename stem.
    copy_tree(src_val_lbl, dst_lbl_val, overwrite=True)

    total = 0
    failed = []
    for img_path in iter_images(src_val_img):
        rel = img_path.relative_to(src_val_img)
        out_path = dst_img_val / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(str(img_path))
            continue

        if args.method == "clahe_gamma":
            out = enhance_clahe_gamma(img, gamma=args.gamma, clip_limit=args.clip_limit, tile_grid_size=args.tile_grid_size)
        elif args.method == "clahe":
            out = enhance_clahe(img, clip_limit=args.clip_limit, tile_grid_size=args.tile_grid_size)
        else:
            out = enhance_gamma(img, gamma=args.gamma)

        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            failed.append(str(img_path))
            continue
        total += 1

    src_yaml = load_yaml(src)
    names = src_yaml.get("names")
    nc = src_yaml.get("nc", len(names) if isinstance(names, (list, dict)) else None)

    out_yaml = {
        "path": str(dst),
        "train": str((src / "images" / "train").resolve()),
        "val": str((dst / "images" / "val").resolve()),
        "test": str((dst / "images" / "val").resolve()),
    }
    if nc is not None:
        out_yaml["nc"] = nc
    if names is not None:
        out_yaml["names"] = names

    yaml_name = "aodraw_enhanced_val.yaml" if "aodraw" in src.name.lower() else "enhanced_val.yaml"
    yaml_path = dst / yaml_name
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, allow_unicode=True, sort_keys=False)

    meta = {
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "source_yaml": src_yaml.get("_source_yaml"),
        "enhanced_images": total,
        "failed_images": failed,
        "method": args.method,
        "gamma": args.gamma,
        "clip_limit": args.clip_limit,
        "tile_grid_size": args.tile_grid_size,
        "yaml": str(yaml_path),
    }
    with (dst / "enhance_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Enhanced val images: {total}")
    print(f"Failed images: {len(failed)}")
    print(f"Output dataset: {dst}")
    print(f"Output yaml: {yaml_path}")


if __name__ == "__main__":
    main()
