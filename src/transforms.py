import monai
import torch

try:
    import config
except ModuleNotFoundError:
    from src import config


class BroadcastToTensord(monai.transforms.MapTransform):
    def __init__(self, keys, source_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key

    def __call__(self, data):
        target_shape = data[self.source_key].shape[-3:]
        for key in self.key_iterator(data):
            data[key] = monai.data.MetaTensor(
                torch.full(target_shape, data[key])[None]
            )
        return data


transforms_dict = {
    "base_model": {
        "base_transforms": monai.transforms.Compose([
            monai.transforms.LoadImaged(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    [config.seg_keys[2]]
                ),
                image_only=False,
                ensure_channel_first=True
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1],
                name="img_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="img_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2]
                )
            ),
            monai.transforms.CropForegroundd(
                keys=["img_AB", "img_C", "seg_C"],
                source_key="img_AB",
            ),
            monai.transforms.Spacingd(
                keys=["img_AB", "img_C", "seg_C"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest"),
            ),
            monai.transforms.ThresholdIntensityd(
                keys="seg_C",
                threshold=1,
                above=False,
                cval=1
            ),
            monai.transforms.AsDiscreted(
                keys="seg_C",
                to_onehot=config.NUM_CLASSES
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=["img_AB", "img_C", "seg_C"],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["img_AB", "img_C", "seg_C"],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="img_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="img_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            )
        ])
    },
    "with_tissue_seg": {
        "base_transforms": monai.transforms.Compose([
            monai.transforms.LoadImaged(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    [config.seg_keys[2]]
                ),
                image_only=False,
                ensure_channel_first=True
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1],
                name="img_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="img_C",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[config.tissue_seg_keys[0], config.tissue_seg_keys[1]],
                name="tissue_seg_AB",
                dim=0
            ),
            monai.transforms.CropForegroundd(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_C"
                ],
                source_key="img_AB",
            ),
            monai.transforms.Spacingd(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_C"
                ],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest", "nearest", "nearest"),
            ),
            monai.transforms.ThresholdIntensityd(
                keys="seg_C",
                threshold=1,
                above=False,
                cval=1
            ),
            monai.transforms.AsDiscreted(
                keys="seg_C",
                to_onehot=config.NUM_CLASSES
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_C"
                ],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest", "nearest", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_C"
                ],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="img_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="img_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "tissue_seg_AB",],
                name="img_tissue_seg_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "tissue_seg_C",],
                name="img_tissue_seg_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    ["img_C", "tissue_seg_AB"]
                )
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "tissue_seg_AB"],
                name="img_tissue_seg_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "tissue_seg_C",],
                name="img_tissue_seg_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    ["img_C", "tissue_seg_AB"]
                )
            )
        ])
    },
    "with_tumor_seg": {
        "base_transforms": monai.transforms.Compose([
            monai.transforms.LoadImaged(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.seg_keys
                ),
                image_only=False,
                ensure_channel_first=True
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1],
                name="img_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="img_C",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[config.seg_keys[0], config.seg_keys[1]],
                name="seg_AB",
                dim=0
            ),
            monai.transforms.CropForegroundd(
                keys=["img_AB", "img_C", "seg_AB", "seg_C"],
                source_key="img_AB",
            ),
            monai.transforms.Spacingd(
                keys=["img_AB", "img_C", "seg_AB", "seg_C"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
            ),
            monai.transforms.ThresholdIntensityd(
                keys=["seg_AB", "seg_C"],
                threshold=1,
                above=False,
                cval=1
            ),
            monai.transforms.CopyItemsd(
                keys="seg_C",
                names="seg_C_input"
            ),
            monai.transforms.AsDiscreted(
                keys="seg_C",
                to_onehot=config.NUM_CLASSES
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=["img_AB", "img_C", "seg_AB", "seg_C", "seg_C_input"],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest", "nearest", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["img_AB", "img_C", "seg_AB", "seg_C", "seg_C_input"],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="img_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="img_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "seg_AB"],
                name="img_tumor_seg_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "seg_C_input"],
                name="img_tumor_seg_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    [
                        "img_C",
                        config.seg_keys[0],
                        config.seg_keys[1],
                        "seg_AB",
                        "seg_C_input"
                    ]
                )
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "seg_AB"],
                name="img_tumor_seg_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "seg_C_input"],
                name="img_tumor_seg_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    [
                        "img_C",
                        config.seg_keys[0],
                        config.seg_keys[1],
                        "seg_AB",
                        "seg_C_input"
                    ]
                )
            )
        ])
    },
    "with_time_diff": {
        "base_transforms": monai.transforms.Compose([
            monai.transforms.LoadImaged(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    [config.seg_keys[2]]
                ),
                image_only=False,
                ensure_channel_first=True
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1],
                name="img_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="img_C",
                dim=0
            ),
            monai.transforms.CropForegroundd(
                keys=["img_AB", "img_C", "seg_C"],
                source_key="img_AB",
            ),
            monai.transforms.Spacingd(
                keys=["img_AB", "img_C", "seg_C"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest"),
            ),
            monai.transforms.ThresholdIntensityd(
                keys="seg_C",
                threshold=1,
                above=False,
                cval=1
            ),
            monai.transforms.AsDiscreted(
                keys="seg_C",
                to_onehot=config.NUM_CLASSES
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=["img_AB", "img_C", "seg_C"],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["img_AB", "img_C", "seg_C"],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="img_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="img_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            BroadcastToTensord(
                keys=["time_diff_AC", "time_diff_BC"],
                source_key="seg_C"
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "time_diff_AC", "time_diff_BC"],
                name="img_time_diff_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "time_diff_AC", "time_diff_BC"],
                name="img_time_diff_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    ["img_C", "time_diff_AC", "time_diff_BC"]
                )
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            BroadcastToTensord(
                keys=["time_diff_AC", "time_diff_BC"],
                source_key="seg_C"
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_AB", "time_diff_AC", "time_diff_BC"],
                name="img_time_diff_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=["img_C", "time_diff_AC", "time_diff_BC"],
                name="img_time_diff_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    ["img_C", "time_diff_AC", "time_diff_BC"]
                )
            )
        ])
    },
    "with_tissue_seg_tumor_seg_time_diff": {
        "base_transforms": monai.transforms.Compose([
            monai.transforms.LoadImaged(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    config.seg_keys
                ),
                image_only=False,
                ensure_channel_first=True
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1],
                name="img_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="img_C",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[config.tissue_seg_keys[0], config.tissue_seg_keys[1]],
                name="tissue_seg_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[config.seg_keys[0], config.seg_keys[1]],
                name="seg_AB",
                dim=0
            ),
            monai.transforms.CropForegroundd(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_AB",
                    "seg_C"
                ],
                source_key="img_AB",
            ),
            monai.transforms.Spacingd(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_AB",
                    "seg_C"
                ],
                pixdim=(1.0, 1.0, 1.0),
                mode=(
                    "bilinear",
                    "bilinear",
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest"
                ),
            ),
            monai.transforms.ThresholdIntensityd(
                keys=["seg_AB", "seg_C"],
                threshold=1,
                above=False,
                cval=1
            ),
            monai.transforms.CopyItemsd(
                keys="seg_C",
                names="seg_C_input"
            ),
            monai.transforms.AsDiscreted(
                keys="seg_C",
                to_onehot=config.NUM_CLASSES
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_AB",
                    "seg_C",
                    "seg_C_input"
                ],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=(
                    "bilinear",
                    "bilinear",
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest",
                    "nearest"
                ),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=[
                    "img_AB",
                    "img_C",
                    "tissue_seg_AB",
                    "tissue_seg_C",
                    "seg_AB",
                    "seg_C",
                    "seg_C_input"
                ],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="img_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="img_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            BroadcastToTensord(
                keys=["time_diff_AC", "time_diff_BC"],
                source_key="seg_C"
            ),
            monai.transforms.ConcatItemsd(
                keys=[
                    "img_AB",
                    "tissue_seg_AB",
                    "seg_AB",
                    "time_diff_AC",
                    "time_diff_BC"
                ],
                name="img_tissue_seg_tumor_seg_time_diff_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[
                    "img_C",
                    "tissue_seg_C",
                    "seg_C_input",
                    "time_diff_AC",
                    "time_diff_BC"
                ],
                name="img_tissue_seg_tumor_seg_time_diff_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    [
                        "img_C",
                        "tissue_seg_AB",
                        config.seg_keys[0],
                        config.seg_keys[1],
                        "seg_AB",
                        "seg_C_input",
                        "time_diff_AC",
                        "time_diff_BC"
                    ]
                )
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(
                keys=["img_AB", "img_C"],
                channel_wise=True
            ),
            BroadcastToTensord(
                keys=["time_diff_AC", "time_diff_BC"],
                source_key="seg_C"
            ),
            monai.transforms.ConcatItemsd(
                keys=[
                    "img_AB",
                    "tissue_seg_AB",
                    "seg_AB",
                    "time_diff_AC",
                    "time_diff_BC"
                ],
                name="img_tissue_seg_tumor_seg_time_diff_AB",
                dim=0
            ),
            monai.transforms.ConcatItemsd(
                keys=[
                    "img_C",
                    "tissue_seg_C",
                    "seg_C_input",
                    "time_diff_AC",
                    "time_diff_BC"
                ],
                name="img_tissue_seg_tumor_seg_time_diff_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=(
                    config.sequence_keys[0] +
                    config.sequence_keys[1] +
                    config.sequence_keys[2] +
                    config.tissue_seg_keys +
                    [
                        "img_C",
                        "tissue_seg_AB",
                        config.seg_keys[0],
                        config.seg_keys[1],
                        "seg_AB",
                        "seg_C_input",
                        "time_diff_AC",
                        "time_diff_BC"
                    ]
                )
            )
        ])
    },
    "compare": monai.transforms.Compose([
        monai.transforms.LoadImaged(
            keys=(
                config.sequence_keys[0] +
                config.sequence_keys[1] +
                [config.tissue_seg_keys[0], config.tissue_seg_keys[1]] +
                config.seg_keys
            ),
            image_only=False,
            ensure_channel_first=True
        ),
        monai.transforms.ConcatItemsd(
            keys=config.sequence_keys[0] + config.sequence_keys[1],
            name="img_AB",
            dim=0
        ),
        monai.transforms.ConcatItemsd(
            keys=[config.tissue_seg_keys[0], config.tissue_seg_keys[1]],
            name="tissue_seg_AB",
            dim=0
        ),
        monai.transforms.ConcatItemsd(
            keys=[config.seg_keys[0], config.seg_keys[1]],
            name="seg_AB",
            dim=0
        ),
        monai.transforms.CropForegroundd(
            keys=["img_AB", "tissue_seg_AB", "seg_AB", "seg_C"],
            source_key="img_AB",
        ),
        monai.transforms.Spacingd(
            keys=["img_AB", "tissue_seg_AB", "seg_AB", "seg_C"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest", "nearest", "nearest"),
        ),
        monai.transforms.ThresholdIntensityd(
            keys=["seg_AB", "seg_C"],
            threshold=1,
            above=False,
            cval=1
        ),
        monai.transforms.AsDiscreted(
            keys="seg_C",
            to_onehot=config.NUM_CLASSES
        ),
        monai.transforms.ScaleIntensityd(
            keys="img_AB",
            channel_wise=True
        ),
        BroadcastToTensord(
            keys=["time_diff_AC", "time_diff_BC"],
            source_key="seg_C"
        ),
        monai.transforms.ConcatItemsd(
            keys=["img_AB", "tissue_seg_AB"],
            name="img_tissue_seg_AB",
            dim=0
        ),
        monai.transforms.ConcatItemsd(
            keys=["img_AB", "seg_AB"],
            name="img_tumor_seg_AB",
            dim=0
        ),
        monai.transforms.ConcatItemsd(
            keys=["img_AB", "time_diff_AC", "time_diff_BC"],
            name="img_time_diff_AB",
            dim=0
        ),
        monai.transforms.ConcatItemsd(
            keys=[
                "img_AB",
                "tissue_seg_AB",
                "seg_AB",
                "time_diff_AC",
                "time_diff_BC"
            ],
            name="img_tissue_seg_tumor_seg_time_diff_AB",
            dim=0
        ),
        monai.transforms.DeleteItemsd(
            keys=(
                config.sequence_keys[0] +
                config.sequence_keys[1] +
                [
                    config.tissue_seg_keys[0],
                    config.tissue_seg_keys[1],
                    "tissue_seg_AB",
                    config.seg_keys[0],
                    config.seg_keys[1],
                    "seg_AB",
                    "time_diff_AC",
                    "time_diff_BC"
                ]
            )
        )
    ])
}
