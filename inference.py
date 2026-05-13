"""Run Mask R-CNN inference on a manga image.

The script loads a trained detector, writes a blended segmentation overlay,
and exports each kept instance as a masked crop.
"""

from __future__ import annotations

import argparse
import glob
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from mmcv.image import imread, imwrite
from mmdet.apis import inference_detector, init_detector


def project_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.name)
    last_number = int(numbers[-1]) if numbers else 0
    return last_number, path.name


def resolve_latest_project():
    project_dirs = sorted(Path("work_dirs").glob("manga_maskrcnn_v*"), key=project_sort_key)
    if not project_dirs:
        raise FileNotFoundError("No project directories found in work_dirs")
    return project_dirs[-1]


PROJECT_NAME = resolve_latest_project().name
DEFAULT_CONFIG = Path("work_dirs") / PROJECT_NAME / "mask_rcnn_manga.py"
DEFAULT_CHECKPOINT = Path("work_dirs") / PROJECT_NAME
DEFAULT_OUTPUT_DIR = Path("out")
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_DEVICE = "cuda:0"
DEFAULT_WORKERS = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLIT_SEGMENT_VALID_DIR = "valid"
SPLIT_SEGMENT_INVALID_DIR = "invalid"

WORKER_MODEL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mask R-CNN inference on a manga image and save an overlay plus panel crops."
    )
    parser.add_argument("img_path", type=str, help="Path to an input image.")
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
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output file or directory for the overlay and panel crops.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum confidence required to keep a detected panel.",
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
        "--split-threshold-saves",
        action="store_true",
        help="Save all instance crops into valid and invalid subfolders based on the score threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of images to process in parallel. Each worker loads its own model.",
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


def collect_input_images(input_path: str) -> list[Path]:
    path = Path(input_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        images = [candidate for candidate in sorted(path.rglob("*")) if candidate.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            return images
        raise FileNotFoundError(f"No supported images found in {path}")

    raise FileNotFoundError(f"Input path does not exist: {path}")


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


def load_inference_image(img_path: str, use_grayscale: bool) -> tuple[np.ndarray, np.ndarray]:
    original_image = imread(img_path)

    if not use_grayscale:
        return original_image, original_image

    grayscale = np.dot(original_image[..., :3], np.array([0.114, 0.587, 0.299], dtype=np.float32))
    grayscale = np.clip(np.rint(grayscale), 0, 255).astype(np.uint8)
    grayscale_image = np.repeat(grayscale[..., None], 3, axis=2)
    return original_image, grayscale_image


def resolve_output_paths(img_path: str, out_path: str | Path | None) -> tuple[Path, Path]:
    image_path = Path(img_path)

    if out_path is None:
        overlay_path = DEFAULT_OUTPUT_DIR / f"{image_path.stem}_segmented.png"
    else:
        candidate = Path(out_path)
        if candidate.suffix:
            overlay_path = candidate
        else:
            overlay_path = candidate / f"{image_path.stem}_segmented.png"

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    segment_dir = overlay_path.with_name(f"{overlay_path.stem}_segments")
    return overlay_path, segment_dir


def resolve_split_segment_dirs(segment_dir: Path) -> tuple[Path, Path]:
    valid_dir = segment_dir / SPLIT_SEGMENT_VALID_DIR
    invalid_dir = segment_dir / SPLIT_SEGMENT_INVALID_DIR
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    return valid_dir, invalid_dir


def initialize_worker(config_path: str, checkpoint_path: str, device: str) -> None:
    global WORKER_MODEL
    WORKER_MODEL = init_detector(config_path, checkpoint_path, device=device)


def print_image_result(
    image_path: Path,
    detected_count: int,
    kept_count: int,
    score_thr: float,
    split_threshold_saves: bool,
) -> None:
    if detected_count == 0:
        print(f"{image_path}: 0 panels detected; saved original image to overlay.")
    elif split_threshold_saves:
        below_count = detected_count - kept_count
        print(
            f"{image_path}: saved {kept_count} valid crop(s) and {below_count} invalid crop(s) at "
            f"threshold {score_thr:.2f}; saved overlay."
        )
    else:
        print(
            f"{image_path}: {kept_count}/{detected_count} panels above score threshold {score_thr:.2f}; "
            f"saved overlay and {kept_count} crop(s)."
        )


def run_jobs_sequentially(
    jobs: list[tuple[str, str | None, float, bool, bool]],
    model,
    score_thr: float,
    split_threshold_saves: bool,
) -> tuple[int, int]:
    total_panels = 0
    total_above_threshold = 0

    for image_path_str, output_path_str, job_score_thr, job_split_threshold_saves, use_grayscale in jobs:
        image_path = Path(image_path_str)
        original_image, inference_image = load_inference_image(image_path_str, use_grayscale)
        result = inference_detector(model, inference_image)

        detected_count, kept_count = save_segmented_outputs(
            image_path_str,
            original_image,
            result,
            job_score_thr,
            output_path_str,
            job_split_threshold_saves,
        )
        total_panels += detected_count
        total_above_threshold += kept_count
        print_image_result(image_path, detected_count, kept_count, score_thr, split_threshold_saves)

    return total_panels, total_above_threshold


def process_image_job(job: tuple[str, str | None, float, bool, bool]) -> tuple[str, int, int]:
    if WORKER_MODEL is None:
        raise RuntimeError("Worker model has not been initialized")

    image_path_str, output_path_str, score_thr, split_threshold_saves, use_grayscale = job
    original_image, inference_image = load_inference_image(image_path_str, use_grayscale)
    result = inference_detector(WORKER_MODEL, inference_image)
    detected_count, kept_count = save_segmented_outputs(
        image_path_str,
        original_image,
        result,
        score_thr,
        output_path_str,
        split_threshold_saves,
    )
    return image_path_str, detected_count, kept_count


def build_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    if len(masks) == 0:
        return overlay

    colors = np.array(
        [
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [230, 25, 75],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
        ],
        dtype=np.float32,
    )
    alpha = 0.45

    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        mask_pixels = mask.astype(bool)
        overlay[mask_pixels] = (overlay[mask_pixels].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(
            np.uint8
        )

    return overlay


def save_segmented_outputs(
    img_path: str,
    original_image: np.ndarray,
    result,
    score_thr: float,
    out_path: str | Path | None,
    split_threshold_saves: bool = False,
) -> tuple[int, int]:
    overlay_path, segment_dir = resolve_output_paths(img_path, out_path)

    instances = getattr(result, "pred_instances", None)
    scores = to_numpy_scores(getattr(instances, "scores", None) if instances is not None else None)
    masks = to_numpy_masks(getattr(instances, "masks", None) if instances is not None else None)

    if scores.size == 0 or masks.size == 0:
        imwrite(original_image, str(overlay_path))
        return 0, 0

    kept_indices = np.flatnonzero(scores >= score_thr)
    kept_masks = masks[kept_indices]

    if split_threshold_saves:
        overlay = original_image if kept_indices.size == 0 else build_overlay(original_image, kept_masks)
        imwrite(overlay, str(overlay_path))
        valid_dir, invalid_dir = resolve_split_segment_dirs(segment_dir)

        for panel_index, mask in enumerate(masks, start=1):
            mask_rows, mask_cols = np.where(mask)
            if mask_rows.size == 0 or mask_cols.size == 0:
                continue

            top, bottom = mask_rows.min(), mask_rows.max() + 1
            left, right = mask_cols.min(), mask_cols.max() + 1

            cropped_image = original_image[top:bottom, left:right]
            cropped_mask = mask[top:bottom, left:right]
            segment = np.full_like(cropped_image, 255)
            segment[cropped_mask] = cropped_image[cropped_mask]

            target_dir = valid_dir if scores[panel_index - 1] >= score_thr else invalid_dir
            segment_path = target_dir / f"{Path(img_path).stem}_panel_{panel_index:02d}.png"
            imwrite(segment, str(segment_path))

        return int(scores.size), int(len(kept_masks))

    if kept_indices.size == 0:
        imwrite(original_image, str(overlay_path))
        return int(scores.size), 0

    overlay = build_overlay(original_image, kept_masks)
    imwrite(overlay, str(overlay_path))
    segment_dir.mkdir(parents=True, exist_ok=True)

    for panel_index, mask in enumerate(kept_masks, start=1):
        mask_rows, mask_cols = np.where(mask)
        if mask_rows.size == 0 or mask_cols.size == 0:
            continue

        top, bottom = mask_rows.min(), mask_rows.max() + 1
        left, right = mask_cols.min(), mask_cols.max() + 1

        cropped_image = original_image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]
        segment = np.full_like(cropped_image, 255)
        segment[cropped_mask] = cropped_image[cropped_mask]

        segment_path = segment_dir / f"{Path(img_path).stem}_panel_{panel_index:02d}.png"
        imwrite(segment, str(segment_path))

    return int(scores.size), int(len(kept_masks))


def build_default_output_for_batch(input_root: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / input_root.name


def resolve_batch_output_root(out_arg: str, input_root: Path) -> Path:
    out_path = Path(out_arg)

    if out_arg == str(DEFAULT_OUTPUT_DIR):
        return build_default_output_for_batch(input_root)

    if out_path.suffix:
        return out_path.parent / out_path.stem

    return out_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.img_path)
    input_images = collect_input_images(args.img_path)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if str(checkpoint_path) != args.checkpoint:
        print(f"Using checkpoint: {checkpoint_path}")

    batch_output_root = resolve_batch_output_root(args.out, input_path) if input_path.is_dir() else None

    jobs: list[tuple[str, str | None, float, bool, bool]] = []
    for image_path in input_images:
        if input_path.is_dir() and batch_output_root is not None:
            relative_path = image_path.relative_to(input_path)
            output_path = batch_output_root / relative_path.parent / f"{image_path.stem}_segmented.png"
        else:
            output_path = args.out

        jobs.append(
            (
                str(image_path),
                str(output_path) if output_path is not None else None,
                args.score_thr,
                args.split_threshold_saves,
                args.grayscale,
            )
        )

    total_panels = 0
    total_above_threshold = 0
    should_parallelize = len(jobs) > 1 and args.workers > 1

    if should_parallelize:
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=(args.config, str(checkpoint_path), args.device),
            ) as executor:
                for image_path_str, detected_count, kept_count in executor.map(process_image_job, jobs, chunksize=1):
                    image_path = Path(image_path_str)
                    total_panels += detected_count
                    total_above_threshold += kept_count
                    print_image_result(
                        image_path, detected_count, kept_count, args.score_thr, args.split_threshold_saves
                    )
        except BrokenProcessPool:
            print(
                "Warning: parallel inference crashed; rerunning sequentially. "
                "If you are using CUDA, keep --workers 1 for stability."
            )
            model = init_detector(args.config, str(checkpoint_path), device=args.device)
            total_panels, total_above_threshold = run_jobs_sequentially(
                jobs,
                model,
                args.score_thr,
                args.split_threshold_saves,
            )
    else:
        model = init_detector(args.config, str(checkpoint_path), device=args.device)
        total_panels, total_above_threshold = run_jobs_sequentially(
            jobs,
            model,
            args.score_thr,
            args.split_threshold_saves,
        )

    if len(input_images) == 1:
        below_threshold = total_panels - total_above_threshold
        print(
            f"Summary: {total_above_threshold} panels above threshold, {below_threshold} below threshold, "
            f"threshold={args.score_thr:.2f}."
        )
    else:
        below_threshold = total_panels - total_above_threshold
        print(
            f"Summary: processed {len(input_images)} image(s); {total_above_threshold} panels above threshold, "
            f"{below_threshold} below threshold, threshold={args.score_thr:.2f}."
        )


if __name__ == "__main__":
    main()
