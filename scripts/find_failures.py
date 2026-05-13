from __future__ import annotations

import argparse
import glob
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import cv2
from mmcv.image import imread, imwrite
from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO


def project_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.name)
    last_number = int(numbers[-1]) if numbers else 0
    return last_number, path.name


def resolve_latest_project() -> Path:
    project_dirs = sorted(Path("work_dirs").glob("manga_maskrcnn_v*"), key=project_sort_key)
    if not project_dirs:
        raise FileNotFoundError("No project directories found in work_dirs")
    return project_dirs[-1]


PROJECT_NAME = resolve_latest_project().name
DEFAULT_CONFIG = Path("work_dirs") / PROJECT_NAME / "mask_rcnn_manga.py"
DEFAULT_CHECKPOINT = Path("work_dirs") / PROJECT_NAME
DEFAULT_DATA_ROOT = Path("manga_panels_yolo_merged")
DEFAULT_OUTPUT_DIR = Path("out") / "find_failures"
DEFAULT_TOP_N = 20
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MATCH_IOU = 0.5
DEFAULT_DEVICE = "cuda:0"
DEFAULT_SPLITS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleEntry:
    split: str
    image_id: int
    file_name: str
    image_path: Path


class AnalysisRow(TypedDict):
    split: str
    image_id: int
    file_name: str
    image_path: str
    gt_count: int
    pred_count: int
    matched_count: int
    false_positives: int
    false_negatives: int
    mean_gt_best_iou: float
    mean_pred_best_iou: float
    precision: float
    recall: float
    f1: float
    failure_score: float
    mean_prediction_score: float
    max_prediction_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on train and val images, rank the worst failures, and save comparison artifacts."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory that contains images/train, images/val, train.json, and val.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the MMDetection config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint path, directory, or glob pattern.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where failure reports and saved images will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of worst samples to save.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum prediction score to keep a detected panel.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU,
        help="IoU threshold used to match predicted masks to ground truth masks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Inference device, such as cuda:0 or cpu.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert input images to grayscale before inference.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=["train", "val"],
        help="Dataset splits to analyze.",
    )
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    last_number = int(numbers[-1]) if numbers else 0
    return last_number, path.name


def resolve_checkpoint_path(checkpoint_spec: str) -> Path:
    candidate = Path(checkpoint_spec)

    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        checkpoints = sorted(candidate.glob("*.pth"), key=checkpoint_sort_key)
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(f"No checkpoint files found in {candidate}")

    matches = [Path(match) for match in glob.glob(checkpoint_spec)]
    if matches:
        return max(matches, key=checkpoint_sort_key)

    raise FileNotFoundError(f"No checkpoint found matching {checkpoint_spec}")


def load_inference_image(img_path: str, use_grayscale: bool) -> tuple[np.ndarray, np.ndarray]:
    original_image = imread(img_path)

    if not use_grayscale:
        return original_image, original_image

    grayscale = np.dot(original_image[..., :3], np.array([0.114, 0.587, 0.299], dtype=np.float32))
    grayscale = np.clip(np.rint(grayscale), 0, 255).astype(np.uint8)
    grayscale_image = np.repeat(grayscale[..., None], 3, axis=2)
    return original_image, grayscale_image


def to_numpy_scores(scores) -> np.ndarray:
    if scores is None:
        return np.empty(0, dtype=np.float32)
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.asarray(scores)
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def to_numpy_masks(masks) -> np.ndarray:
    if masks is None:
        return np.empty((0, 0, 0), dtype=bool)
    if hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    elif hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    elif hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    else:
        masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    return masks.astype(bool, copy=False)


def collect_samples(data_root: Path, split: str) -> tuple[COCO, list[SampleEntry]]:
    ann_file = data_root / f"{split}.json"
    image_dir = data_root / "images" / split

    if not ann_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    coco = COCO(str(ann_file))
    samples: list[SampleEntry] = []

    for image_id in sorted(coco.getImgIds()):
        image_info = coco.loadImgs([image_id])[0]
        file_name = image_info["file_name"]
        image_path = image_dir / file_name
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        samples.append(
            SampleEntry(
                split=split,
                image_id=int(image_id),
                file_name=file_name,
                image_path=image_path,
            )
        )

    return coco, samples


def load_ground_truth_masks(coco: COCO, image_id: int) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)
    masks = []

    for ann in anns:
        if int(ann.get("iscrowd", 0)):
            continue
        masks.append(coco.annToMask(ann).astype(bool))

    if not masks:
        image_info = coco.loadImgs([image_id])[0]
        return np.empty((0, int(image_info["height"]), int(image_info["width"])), dtype=bool)

    return np.stack(masks, axis=0)


def extract_prediction_masks(result, score_thr: float) -> tuple[np.ndarray, np.ndarray]:
    instances = getattr(result, "pred_instances", None)
    scores = to_numpy_scores(getattr(instances, "scores", None) if instances is not None else None)
    masks = to_numpy_masks(getattr(instances, "masks", None) if instances is not None else None)

    if scores.size == 0 or masks.size == 0:
        return np.empty((0, 0, 0), dtype=bool), np.empty(0, dtype=np.float32)

    keep = scores >= score_thr
    return masks[keep], scores[keep]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)

    matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            matrix[pred_index, gt_index] = mask_iou(pred_mask, gt_mask)
    return matrix


def greedy_match(iou_matrix: np.ndarray, match_iou: float) -> list[tuple[int, int, float]]:
    if iou_matrix.size == 0:
        return []

    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_index in range(iou_matrix.shape[0]):
        for gt_index in range(iou_matrix.shape[1]):
            iou = float(iou_matrix[pred_index, gt_index])
            if iou >= match_iou:
                candidate_pairs.append((iou, pred_index, gt_index))

    candidate_pairs.sort(reverse=True)
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, pred_index, gt_index in candidate_pairs:
        if pred_index in used_preds or gt_index in used_gts:
            continue
        used_preds.add(pred_index)
        used_gts.add(gt_index)
        matches.append((pred_index, gt_index, iou))

    return matches


def build_overlay(image: np.ndarray, masks: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    if len(masks) == 0:
        return overlay

    color_array = np.array(color, dtype=np.float32)
    for mask in masks:
        mask_pixels = mask.astype(bool)
        overlay[mask_pixels] = (overlay[mask_pixels].astype(np.float32) * (1.0 - alpha) + color_array * alpha).astype(
            np.uint8
        )

    return overlay


def build_difference_overlay(
    image: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    matches: list[tuple[int, int, float]],
) -> np.ndarray:
    overlay = image.copy()
    matched_pred_indices = {pred_index for pred_index, _, _ in matches}
    matched_gt_indices = {gt_index for _, gt_index, _ in matches}

    for pred_index, mask in enumerate(pred_masks):
        color = (60, 180, 75) if pred_index in matched_pred_indices else (245, 130, 48)
        mask_pixels = mask.astype(bool)
        overlay[mask_pixels] = (
            overlay[mask_pixels].astype(np.float32) * 0.55 + np.array(color, dtype=np.float32) * 0.45
        ).astype(np.uint8)

    for gt_index, mask in enumerate(gt_masks):
        if gt_index in matched_gt_indices:
            continue
        mask_pixels = mask.astype(bool)
        overlay[mask_pixels] = (
            overlay[mask_pixels].astype(np.float32) * 0.55 + np.array((0, 130, 200), dtype=np.float32) * 0.45
        ).astype(np.uint8)

    return overlay


def concatenate_images(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]

    heights = [image.shape[0] for image in images]
    target_height = max(heights)
    resized_images = []
    for image in images:
        if image.shape[0] == target_height:
            resized_images.append(image)
            continue
        scale = target_height / image.shape[0]
        target_width = max(1, int(round(image.shape[1] * scale)))
        resized_images.append(cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR))

    return np.concatenate(resized_images, axis=1)


def summarize_sample(
    sample: SampleEntry,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    matches: list[tuple[int, int, float]],
    iou_matrix: np.ndarray,
) -> AnalysisRow:
    gt_count = int(len(gt_masks))
    pred_count = int(len(pred_masks))
    matched_count = int(len(matches))
    false_positives = pred_count - matched_count
    false_negatives = gt_count - matched_count

    gt_best_iou = float(iou_matrix.max(axis=0).mean()) if gt_count and pred_count else (1.0 if gt_count == 0 else 0.0)
    pred_best_iou = (
        float(iou_matrix.max(axis=1).mean()) if gt_count and pred_count else (1.0 if pred_count == 0 else 0.0)
    )

    precision = matched_count / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
    recall = matched_count / gt_count if gt_count else (1.0 if pred_count == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    failure_score = float(false_positives + false_negatives + (1.0 - gt_best_iou))

    return {
        "split": sample.split,
        "image_id": sample.image_id,
        "file_name": sample.file_name,
        "image_path": str(sample.image_path),
        "gt_count": gt_count,
        "pred_count": pred_count,
        "matched_count": matched_count,
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "mean_gt_best_iou": gt_best_iou,
        "mean_pred_best_iou": pred_best_iou,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "failure_score": failure_score,
        "mean_prediction_score": float(pred_scores.mean()) if pred_scores.size else 0.0,
        "max_prediction_score": float(pred_scores.max()) if pred_scores.size else 0.0,
    }


def save_sample_artifacts(
    sample_dir: Path,
    image_path: Path,
    original_image: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    matches: list[tuple[int, int, float]],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_overlay = build_overlay(original_image, pred_masks, (60, 180, 75))
    gt_overlay = build_overlay(original_image, gt_masks, (230, 25, 75))
    diff_overlay = build_difference_overlay(original_image, pred_masks, gt_masks, matches)
    comparison = concatenate_images([pred_overlay, gt_overlay, diff_overlay])

    imwrite(original_image, str(sample_dir / f"{image_path.stem}_original.png"))
    imwrite(pred_overlay, str(sample_dir / f"{image_path.stem}_prediction_overlay.png"))
    imwrite(gt_overlay, str(sample_dir / f"{image_path.stem}_truth_overlay.png"))
    imwrite(diff_overlay, str(sample_dir / f"{image_path.stem}_difference_overlay.png"))
    imwrite(comparison, str(sample_dir / f"{image_path.stem}_comparison.png"))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if str(checkpoint_path) != args.checkpoint:
        print(f"Using checkpoint: {checkpoint_path}")

    model = init_detector(args.config, str(checkpoint_path), device=args.device)

    all_rows: list[AnalysisRow] = []
    sample_lookup: dict[tuple[str, int], SampleEntry] = {}
    coco_lookup: dict[str, COCO] = {}

    for split in args.splits:
        coco, samples = collect_samples(data_root, split)
        coco_lookup[split] = coco
        print(f"Processing {split}: {len(samples)} image(s)")

        for sample in samples:
            sample_lookup[(sample.split, sample.image_id)] = sample
            original_image, inference_image = load_inference_image(str(sample.image_path), args.grayscale)
            result = inference_detector(model, inference_image)
            pred_masks, pred_scores = extract_prediction_masks(result, args.score_thr)
            gt_masks = load_ground_truth_masks(coco, sample.image_id)
            iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
            matches = greedy_match(iou_matrix, args.match_iou)
            row = summarize_sample(sample, gt_masks, pred_masks, pred_scores, matches, iou_matrix)
            all_rows.append(row)

    ranked_rows: list[AnalysisRow] = sorted(
        all_rows,
        key=lambda row: (
            -row["failure_score"],
            -row["false_positives"] - row["false_negatives"],
            row["mean_gt_best_iou"],
        ),
    )

    csv_path = out_dir / "failure_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "image_id",
            "file_name",
            "image_path",
            "gt_count",
            "pred_count",
            "matched_count",
            "false_positives",
            "false_negatives",
            "mean_gt_best_iou",
            "mean_pred_best_iou",
            "precision",
            "recall",
            "f1",
            "failure_score",
            "mean_prediction_score",
            "max_prediction_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    top_rows = ranked_rows[: max(0, args.top_n)]
    top_lookup: dict[tuple[str, int], int] = {
        (row["split"], row["image_id"]): rank for rank, row in enumerate(top_rows, start=1)
    }

    for row in top_rows:
        key = (row["split"], row["image_id"])
        sample = sample_lookup[key]
        coco = coco_lookup[sample.split]
        original_image, inference_image = load_inference_image(str(sample.image_path), args.grayscale)
        result = inference_detector(model, inference_image)
        pred_masks, pred_scores = extract_prediction_masks(result, args.score_thr)
        gt_masks = load_ground_truth_masks(coco, sample.image_id)
        iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
        matches = greedy_match(iou_matrix, args.match_iou)

        rank = top_lookup[key]
        sample_dir = out_dir / f"rank_{rank:03d}_{sample.split}_{sample.image_path.stem}"
        save_sample_artifacts(sample_dir, sample.image_path, original_image, pred_masks, gt_masks, matches)

        summary_path = sample_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **row,
                    "recomputed_prediction_count": len(pred_masks),
                    "recomputed_mean_prediction_score": float(pred_scores.mean()) if pred_scores.size else 0.0,
                },
                handle,
                indent=2,
            )

    print(f"Saved ranked report to {csv_path}")
    print(f"Saved top {len(top_rows)} failure sample(s) under {out_dir}")

    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:02d}. {row['split']}/{row['file_name']}: score={row['failure_score']:.3f}, "
            f"fp={row['false_positives']}, fn={row['false_negatives']}, f1={row['f1']:.3f}, "
            f"gt_iou={row['mean_gt_best_iou']:.3f}"
        )


if __name__ == "__main__":
    main()
