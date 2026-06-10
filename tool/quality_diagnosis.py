from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def get_images(data_yaml: str | Path, split: str) -> List[Path]:
    data_yaml = Path(data_yaml).resolve()
    data = load_yaml(data_yaml)
    yaml_dir = data_yaml.parent
    root = resolve_path(yaml_dir, data.get("path", yaml_dir))
    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"Split not found in dataset yaml: {split}")
    split_path = resolve_path(root, split_value)

    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as f:
            return [resolve_path(root, x.strip()) for x in f if x.strip()]

    images: List[Path] = []
    for ext in IMG_EXTS:
        images.extend(split_path.rglob(f"*{ext}"))
        images.extend(split_path.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def label_path_from_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def read_yolo_label(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) >= 5:
                rows.append((int(float(sp[0])), *map(float, sp[1:5])))
    return rows


def image_quality(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    brightness = float(g.mean())
    contrast = float(g.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low = cv2.GaussianBlur(g, (5, 5), 0)
    noise = float(np.abs(g - low).mean())
    return {
        "brightness": brightness,
        "darkness": 1.0 - brightness,
        "contrast": contrast,
        "low_contrast": 1.0 - min(contrast / 0.25, 1.0),
        "blur_score": lap_var,
        "noise_score": noise,
    }


def brightness_bin(v: float) -> str:
    if v >= 0.45:
        return "easy_light"
    if v >= 0.30:
        return "medium_light"
    if v >= 0.18:
        return "hard_light"
    return "extreme_dark"


def scale_bin(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create image/object-level low-quality statistics for a YOLO dataset.")
    parser.add_argument("--data", required=True, help="Dataset yaml path")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--name", required=True, help="Output file prefix")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = get_images(args.data, args.split)

    image_rows = []
    object_rows = []
    for img_path in tqdm(images, desc=args.name):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        q = image_quality(img)
        bbin = brightness_bin(q["brightness"])
        labels = read_yolo_label(label_path_from_image(img_path))
        image_rows.append({"image": str(img_path), "num_objects": len(labels), "brightness_bin": bbin, **q})
        for cls, x, y, w, h in labels:
            area = w * h
            small_score = 1.0 - min(area / 0.05, 1.0)
            q_obj = 0.6 * q["darkness"] + 0.25 * small_score + 0.15 * q["low_contrast"]
            object_rows.append({
                "image": str(img_path),
                "class_id": cls,
                "area_norm": area,
                "scale_bin": scale_bin(area),
                "brightness_bin": bbin,
                "q_obj": max(0.0, min(1.0, float(q_obj))),
                **q,
            })

    df_img = pd.DataFrame(image_rows)
    df_obj = pd.DataFrame(object_rows)
    df_img.to_csv(out / f"{args.name}_image_quality.csv", index=False)
    df_obj.to_csv(out / f"{args.name}_object_quality.csv", index=False)

    if not df_obj.empty:
        group = df_obj.groupby(["brightness_bin", "scale_bin"]).agg(
            num_objects=("image", "count"),
            mean_q_obj=("q_obj", "mean"),
            mean_darkness=("darkness", "mean"),
            mean_area=("area_norm", "mean"),
        ).reset_index()
        group.to_csv(out / f"{args.name}_quality_groups.csv", index=False)

    print(f"Saved quality diagnosis files to: {out}")


if __name__ == "__main__":
    main()
