_base_ = ["mmdet::mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py"]

NAME = "manga_maskrcnn_v16"
EPOCHS = 48

# ========================
# DATASET
# ========================
data_root = r"C:\Users\lucas\Documents\GitHub\manga-segment-train\manga_panels_yolo_merged"

metainfo = {"classes": ("panel",), "palette": [(220, 20, 60)]}

fp16 = dict(loss_scale=512.0)

train_dataloader = dict(
    batch_size=24,
    num_workers=8,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img="images/train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", scale=(1344, 1344), keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(
                type="Albu",
                transforms=[
                    dict(type="SafeRotate", limit=30, border_mode=0, p=0.5),
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
            dict(type="Resize", scale=(1344, 1344), keep_ratio=True),
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
            dict(type="Resize", scale=(1344, 1344), keep_ratio=True),
            dict(type="PackDetInputs"),
        ],
    ),
)

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "/val.json", metric=["bbox", "segm"])

test_evaluator = val_evaluator

# ========================
# MODEL (Standalone PointRend)
# ========================
model = dict(
    type="PointRend",
    roi_head=dict(
        type="PointRendRoIHead",
        mask_roi_extractor=dict(
            type="GenericRoIExtractor",
            aggregation="concat",
            roi_layer=dict(_delete_=True, type="SimpleRoIAlign", output_size=14),
            out_channels=256,
            featmap_strides=[4],
        ),
        mask_head=dict(
            _delete_=True,
            type="CoarseMaskHead",
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
        point_head=dict(
            type="MaskPointHead",
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=1,
            coarse_pred_each_layer=True,
            loss_point=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75,
        )
    ),
    test_cfg=dict(rcnn=dict(subdivision_steps=5, subdivision_num_points=28 * 28, scale_factor=2)),
)

# ========================
# OPTIMIZATION
# ========================
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=EPOCHS, val_interval=2)

WARMUP_STEPS = 500

# Define milestones as fractions of EPOCHS
MILESTONE_1 = int(EPOCHS * 0.6)
MILESTONE_2 = int(EPOCHS * 0.9)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=WARMUP_STEPS),
    dict(type="MultiStepLR", milestones=[MILESTONE_1, MILESTONE_2], gamma=0.25, by_epoch=True),
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
