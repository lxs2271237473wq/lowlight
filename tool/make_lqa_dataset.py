#LQA 数据集生成脚本
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(base, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return (Path(base) / p).resolve()


def get_images(data_yaml, split="train"):
    data_yaml = Path(data_yaml).resolve()
    data = read_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    root = data.get("path", yaml_dir)
    root = resolve_path(yaml_dir, root)

    split_value = data.get(split)
    split_path = resolve_path(root, split_value)

    if split_path.is_file():
        with open(split_path, "r", encoding="utf-8") as f:
            images = [x.strip() for x in f.readlines() if x.strip()]
        return [resolve_path(root, x) for x in images], data, root

    images = []
    for ext in IMG_EXTS:
        images.extend(split_path.rglob(f"*{ext}"))

    return sorted(set(images)), data, root


def image_to_label_path(img_path):
    parts = list(Path(img_path).parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return Path(img_path).with_suffix(".txt")


def load_labels(label_path):
    rows = []
    label_path = Path(label_path)
    if not label_path.exists():
        return rows

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            cls = int(float(sp[0]))
            x, y, w, h = map(float, sp[1:5])
            rows.append((cls, x, y, w, h))
    return rows


def calc_brightness(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(gray.mean())


def get_repeat_factor(img_path, max_repeat=3):
    brightness = calc_brightness(img_path)
    labels = load_labels(image_to_label_path(img_path))

    has_small = False
    has_tiny = False

    for _, _, _, w, h in labels:
        area = w * h
        if area < 0.01:
            has_small = True
        if area < 0.005:
            has_tiny = True

    is_dark = brightness < 0.30
    is_extreme_dark = brightness < 0.18

    repeat = 1

    if has_small and is_dark:
        repeat += 1

    if has_tiny and is_extreme_dark:
        repeat += 1

    return min(repeat, max_repeat), brightness, len(labels), has_small, is_dark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max_repeat", type=int, default=3)
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images, data, root = get_images(data_yaml, "train")

    weighted_train = []
    stats_rows = []

    for img in train_images:
        repeat, brightness, num_labels, has_small, is_dark = get_repeat_factor(
            img, max_repeat=args.max_repeat
        )

        for _ in range(repeat):
            weighted_train.append(str(img))

        stats_rows.append(
            f"{img},{repeat},{brightness:.6f},{num_labels},{int(has_small)},{int(is_dark)}"
        )

    train_txt = out_dir / f"{args.name}_train_lqa.txt"
    stats_csv = out_dir / f"{args.name}_lqa_sampling_stats.csv"
    new_yaml = out_dir / f"{args.name}_lqa.yaml"

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(weighted_train) + "\n")

    with open(stats_csv, "w", encoding="utf-8") as f:
        f.write("image,repeat,brightness,num_labels,has_small,is_dark\n")
        f.write("\n".join(stats_rows) + "\n")

    new_data = dict(data)
    new_data["path"] = str(root)
    new_data["train"] = str(train_txt.resolve())
    new_data["val"] = data["val"]

    if "test" in data:
        new_data["test"] = data["test"]

    write_yaml(new_yaml, new_data)

    print("LQA dataset generated.")
    print(f"original train images: {len(train_images)}")
    print(f"weighted train lines: {len(weighted_train)}")
    print(f"train txt: {train_txt}")
    print(f"stats csv: {stats_csv}")
    print(f"yaml: {new_yaml}")


if __name__ == "__main__":
    main()
