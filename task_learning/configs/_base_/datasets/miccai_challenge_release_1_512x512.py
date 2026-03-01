miccai_1_type = "Endovis2018Dataset"
miccai_1_root = "data/endovis2018/miccai_challenge_release_1/"
miccai_1_crop_size = (512, 512)
miccai_1_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 1024)),
    dict(type="RandomCrop", crop_size=miccai_1_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
miccai_1_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_miccai_1 = dict(
    type=miccai_1_type,
    data_root=miccai_1_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=miccai_1_train_pipeline,
)
val_miccai_1 = dict(
    type=miccai_1_type,
    data_root=miccai_1_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=miccai_1_test_pipeline,
)
