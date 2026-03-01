miccai_2_type = "Endovis2018Dataset"
miccai_2_root = "data/endovis2018/miccai_challenge_release_2/"
miccai_2_crop_size = (512, 512)
miccai_2_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 1024)),
    dict(type="RandomCrop", crop_size=miccai_2_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
miccai_2_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_miccai_2 = dict(
    type=miccai_2_type,
    data_root=miccai_2_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    ignore_index=11,
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=miccai_2_train_pipeline,
)
val_miccai_2 = dict(
    type=miccai_2_type,
    data_root=miccai_2_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    ignore_index=11,
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=miccai_2_test_pipeline,
)
