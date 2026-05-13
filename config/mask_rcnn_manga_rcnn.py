_base_ = ["mmdet::mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py"]

NAME = "manga_maskrcnn_v10"
EPOCHS = 24

# ========================
# DATASET
# ========================
data_root = r"C:\Users\lucas\Documents\GitHub\manga-segment-train\manga_panels_yolo_merged"

metainfo = {"classes": ("panel",), "palette": [(220, 20, 60)]}

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img="images/train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(
                type="Albu",
                transforms=[
                    dict(type="SafeRotate", limit=30, border_mode=0, value=0, p=0.5),
                    dict(type="RandomBrightnessContrast", brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    dict(type="HueSaturationValue", hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ],
                # bbox_params ensures bounding boxes are updated correctly
                bbox_params=dict(
                    type="BboxParams",
                    format="pascal_voc",
                    label_fields=["gt_bboxes_labels"],
                    min_visibility=0.3,
                ),
                keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
            ),
            dict(type="PackDetInputs"),
        ],
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img="images/val/"),
        metainfo=metainfo,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="PackDetInputs"),
        ],
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img="images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
            dict(type="PackDetInputs"),
        ],
    ),
)

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "/val.json", metric=["bbox", "segm"])

test_evaluator = val_evaluator

# ========================
# MODEL (CRITICAL TUNING)
# ========================
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            type="FCNMaskHead",
            num_classes=1,
            in_channels=256,
            conv_out_channels=256,
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type="nms", iou_threshold=0.4),
            max_per_img=500,
        )
    ),
)

# ========================
# OPTIMIZATION
# ========================
optim_wrapper = dict(
    _delete_=True,  # ← Forces removal of inherited optim_wrapper
    type="OptimWrapper",  # ← Explicitly state the wrapper type
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
    ),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=EPOCHS, val_interval=2)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", by_epoch=True, T_max=EPOCHS),
]

# ========================
# RUNTIME
# ========================
default_hooks = dict(
    checkpoint=dict(
        interval=10000,
        save_best="coco/segm_mAP",
        rule="greater",
        save_last=False,
    ),
    logger=dict(interval=50),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="manga-maskrcnn", name=NAME)),
]

visualizer = dict(type="DetLocalVisualizer", name="visualizer", vis_backends=vis_backends)

log_level = "INFO"
work_dir = f"./work_dirs/{NAME}"
