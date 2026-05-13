from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATA_ROOT = Path("manga_panels_yolo_merged")
DEFAULT_OUTPUT_DIR = Path("out") / "dataset_visualization"
DEFAULT_SPLITS = ("train", "val")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render dataset images with YOLO polygon labels overlaid for quick inspection."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory containing images/ and labels/ subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where rendered images will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=["train", "val"],
        help="Dataset splits to render.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to render per split. Use 0 for all images.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Transparency used for label fills.",
    )
    return parser.parse_args()


def find_image_path(image_dir: Path, stem: str) -> Path:
    exact_match = image_dir / stem
    if exact_match.exists():
        return exact_match

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    matches = sorted(image_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No image found for label stem {stem} in {image_dir}")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def parse_label_file(label_path: Path) -> list[list[np.ndarray]]:
    polygons: list[list[np.ndarray]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Invalid polygon label in {label_path}: {line}")

            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
            if len(coords) % 2 != 0:
                raise ValueError(f"Odd number of coordinates in {label_path}: {line}")

            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            polygons.append([class_id, points])

    return polygons


def draw_label_overlay(image: np.ndarray, labels: list[list[np.ndarray]], alpha: float) -> np.ndarray:
    overlay = image.copy()
    if not labels:
        return overlay

    palette = [
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (230, 25, 75),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
    ]

    height, width = image.shape[:2]
    for index, (class_id, points) in enumerate(labels, start=1):
        color = palette[class_id % len(palette)]
        polygon = np.round(points * np.array([width, height], dtype=np.float32)).astype(np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0.0)

        cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)

        x, y = polygon[0]
        label_text = f"panel {index}"
        text_origin = (int(x), max(20, int(y) - 6))
        cv2.putText(
            overlay,
            label_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay


def render_split(data_root: Path, split: str, out_dir: Path, limit: int, alpha: float) -> None:
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    split_out_dir = out_dir / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))
    if limit > 0:
        label_files = label_files[:limit]

    print(f"Rendering {split}: {len(label_files)} label file(s)")

    for label_path in label_files:
        stem = label_path.stem
        image_path = find_image_path(image_dir, stem)
        image = load_image(image_path)
        labels = parse_label_file(label_path)
        overlay = draw_label_overlay(image, labels, alpha)

        output_path = split_out_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(output_path), overlay)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        render_split(data_root, split, out_dir, args.limit, args.alpha)

    print(f"Saved overlays to {out_dir}")


if __name__ == "__main__":
    main()
