import os
import json
import cv2


def webp_to_jpg(webp_path: str) -> str:
    jpg_path = webp_path.replace(".webp", ".jpg")
    img = cv2.imread(webp_path)
    cv2.imwrite(jpg_path, img)
    return jpg_path


def yolo_to_coco(image_dir: str, label_dir: str, output_json: str):
    images = []
    annotations = []
    ann_id = 1

    for img_id, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith(".webp"):
            img_path = webp_to_jpg(os.path.join(image_dir, filename))
        elif not filename.endswith((".jpg", ".png")):
            continue
        else:
            img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        images.append({"id": img_id, "file_name": filename, "height": h, "width": w})

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                coords = parts[1:]

                # Convert normalized → pixel coords
                poly = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * w
                    y = coords[i + 1] * h
                    poly.extend([x, y])

                # bbox
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "segmentation": [poly],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "iscrowd": 0,
                    }
                )

                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "panel"}]}

    with open(output_json, "w") as f:
        json.dump(coco, f)


BASE_PATH = r"C:\Users\lucas\Documents\GitHub\manga-segment-train\manga_panels_yolo_merged"
yolo_to_coco(
    os.path.join(BASE_PATH, "images", "val"),
    os.path.join(BASE_PATH, "labels", "val"),
    os.path.join(BASE_PATH, "val.json"),
)
yolo_to_coco(
    os.path.join(BASE_PATH, "images", "train"),
    os.path.join(BASE_PATH, "labels", "train"),
    os.path.join(BASE_PATH, "train.json"),
)
