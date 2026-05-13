from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("manga_panels_yolo_merged")
DEFAULT_SPLITS = ("train", "val")


@dataclass
class SplitReport:
    split: str
    image_count: int
    annotation_count: int
    missing_images: list[str]
    images_without_annotations: list[str]
    orphan_annotations: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manga_panels_yolo_merged train/val COCO JSON files against the image folders."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root containing train.json, val.json, and images/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=["train", "val"],
        help="Dataset splits to validate.",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing dataset JSON: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")

    return data


def validate_split(data_root: Path, split: str) -> SplitReport:
    json_path = data_root / f"{split}.json"
    image_dir = data_root / "images" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    data = load_json(json_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not isinstance(images, list):
        raise ValueError(f"Expected 'images' to be a list in {json_path}")
    if not isinstance(annotations, list):
        raise ValueError(f"Expected 'annotations' to be a list in {json_path}")

    image_by_id: dict[int, dict[str, Any]] = {}
    invalid_images: list[str] = []

    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"Invalid image entry in {json_path}: {image!r}")

        image_id = image.get("id")
        file_name = image.get("file_name")

        if image_id is None or file_name is None:
            invalid_images.append(str(image))
            continue

        image_by_id[int(image_id)] = image

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    orphan_annotations: list[str] = []

    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Invalid annotation entry in {json_path}: {annotation!r}")

        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")

        if image_id not in image_by_id:
            orphan_annotations.append(f"annotation {annotation_id} -> missing image_id {image_id}")
            continue

        annotations_by_image[int(image_id)].append(annotation)

    missing_images: list[str] = []
    images_without_annotations: list[str] = []

    for image_id, image in image_by_id.items():
        file_name = str(image["file_name"])
        image_path = image_dir / file_name
        if not image_path.exists():
            missing_images.append(file_name)

        if not annotations_by_image.get(image_id):
            images_without_annotations.append(file_name)

    if invalid_images:
        orphan_annotations.append(f"invalid image metadata for {len(invalid_images)} image entry(ies)")

    return SplitReport(
        split=split,
        image_count=len(image_by_id),
        annotation_count=len(annotations),
        missing_images=missing_images,
        images_without_annotations=images_without_annotations,
        orphan_annotations=orphan_annotations,
    )


def print_list(title: str, values: list[str], limit: int = 20) -> None:
    if not values:
        print(f"{title}: none")
        return

    print(f"{title}: {len(values)}")
    for value in values[:limit]:
        print(f"  - {value}")
    if len(values) > limit:
        print(f"  - ... and {len(values) - limit} more")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {data_root}")

    overall_issues = 0

    for split in args.splits:
        report = validate_split(data_root, split)

        print(f"[{split}] images: {report.image_count}, annotations: {report.annotation_count}")
        print_list("missing images", report.missing_images)
        print_list("images without annotations", report.images_without_annotations)
        print_list("orphan annotations", report.orphan_annotations)

        split_issues = (
            len(report.missing_images) + len(report.images_without_annotations) + len(report.orphan_annotations)
        )
        if split_issues == 0:
            print(f"[{split}] OK")
        else:
            overall_issues += split_issues
        print()

    if overall_issues == 0:
        print("Dataset validation passed.")
        return 0

    print("Dataset validation found issues.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
