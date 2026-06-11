#分组召回率诊断脚本
from pathlib import Path
import argparse
import csv
import yaml
import cv2
import numpy as np
import pandas as pd


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return (Path(base) / p).resolve()


def get_images(data_yaml, split="val"):
    data_yaml = Path(data_yaml).resolve()
    data = read_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    root = data.get("path", yaml_dir)
    root = resolve_path(yaml_dir, root)

    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"{data_yaml} does not contain split: {split}")

    split_path = resolve_path(root, split_value)

    if split_path.is_file():
        with open(split_path, "r", encoding="utf-8") as f:
            images = [x.strip() for x in f.readlines() if x.strip()]
        images = [resolve_path(root, x) for x in images]
    else:
        images = []
        for ext in IMG_EXTS:
            images.extend(split_path.rglob(f"*{ext}"))

    return sorted(set(images))


def image_to_label_path(img_path):
    parts = list(Path(img_path).parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return Path(img_path).with_suffix(".txt")


def load_gt(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            cls = int(float(sp[0]))
            x, y, w, h = map(float, sp[1:5])
            rows.append((cls, x, y, w, h))
    return rows


def load_pred(path, conf_thr=0.001):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            cls = int(float(sp[0]))
            x, y, w, h = map(float, sp[1:5])
            conf = float(sp[5]) if len(sp) >= 6 else 1.0
            if conf >= conf_thr:
                rows.append((cls, x, y, w, h, conf))
    return rows


def xywh_to_xyxy(x, y, w, h):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-9)


def calc_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    brightness = float(gray_f.mean())
    darkness = 1.0 - brightness
    contrast = float(gray_f.std())
    low_contrast = 1.0 - min(contrast / 0.25, 1.0)

    return brightness, darkness, contrast, low_contrast


def brightness_bin(brightness):
    if brightness >= 0.45:
        return "easy_light"
    if brightness >= 0.30:
        return "medium_light"
    if brightness >= 0.18:
        return "hard_light"
    return "extreme_dark"


def scale_bin(area):
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def pred_path_for_image(pred_dir, img_path):
    return Path(pred_dir) / f"{Path(img_path).stem}.txt"


def match_one_image(gts, preds, iou_thr=0.5):
    pred_objs = []
    for cls, x, y, w, h, conf in preds:
        pred_objs.append({
            "cls": cls,
            "box": xywh_to_xyxy(x, y, w, h),
            "conf": conf,
            "used": False,
        })

    results = []

    for cls, x, y, w, h in gts:
        gt_box = xywh_to_xyxy(x, y, w, h)

        best_iou = 0.0
        best_conf = 0.0
        best_idx = -1

        for idx, p in enumerate(pred_objs):
            if p["used"]:
                continue
            if p["cls"] != cls:
                continue

            cur_iou = iou(gt_box, p["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_conf = p["conf"]
                best_idx = idx

        matched = best_iou >= iou_thr

        if matched and best_idx >= 0:
            pred_objs[best_idx]["used"] = True

        results.append({
            "matched": int(matched),
            "best_iou": best_iou,
            "best_conf": best_conf if matched else 0.0,
        })

    return results


def summarize(df, group_col):
    rows = []

    for group, g in df.groupby(group_col):
        n = len(g)
        matched = int(g["matched"].sum())
        recall = matched / n if n else 0.0

        rows.append({
            "group_by": group_col,
            "group": group,
            "num_objects": n,
            "matched": matched,
            "missed": n - matched,
            "recall_iou50": recall,
            "mean_iou": float(g["best_iou"].mean()),
            "mean_conf": float(g["best_conf"].mean()),
            "mean_darkness": float(g["darkness"].mean()),
            "mean_area": float(g["area"].mean()),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="runs/diagnosis")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = get_images(args.data, args.split)

    object_rows = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        brightness, darkness, contrast, low_contrast = calc_quality(img)
        bbin = brightness_bin(brightness)

        gt_path = image_to_label_path(img_path)
        pred_path = pred_path_for_image(args.pred_dir, img_path)

        gts = load_gt(gt_path)
        preds = load_pred(pred_path, args.conf)

        matched = match_one_image(gts, preds, args.iou)

        for idx, gt in enumerate(gts):
            cls, x, y, w, h = gt
            area = w * h
            sbin = scale_bin(area)
            m = matched[idx] if idx < len(matched) else {
                "matched": 0,
                "best_iou": 0.0,
                "best_conf": 0.0,
            }

            object_rows.append({
                "image": str(img_path),
                "class_id": cls,
                "brightness_bin": bbin,
                "scale_bin": sbin,
                "combined_bin": f"{bbin}_{sbin}",
                "is_small_dark": int(sbin == "small" and bbin in ["hard_light", "extreme_dark"]),
                "brightness": brightness,
                "darkness": darkness,
                "contrast": contrast,
                "low_contrast": low_contrast,
                "area": area,
                **m,
            })

    df = pd.DataFrame(object_rows)

    obj_path = out / f"{args.name}_object_eval.csv"
    group_path = out / f"{args.name}_group_eval.csv"

    df.to_csv(obj_path, index=False)

    summary_rows = []
    for col in ["brightness_bin", "scale_bin", "combined_bin", "is_small_dark"]:
        summary_rows.extend(summarize(df, col))

    pd.DataFrame(summary_rows).to_csv(group_path, index=False)

    print(f"Saved object eval: {obj_path}")
    print(f"Saved group eval: {group_path}")


if __name__ == "__main__":
    main()
