import monai

try:
    import config
except ModuleNotFoundError:
    from src import config


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
                name="imgs_AB",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1]
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="imgs_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(keys=config.sequence_keys[2]),
            monai.transforms.CropForegroundd(
                keys=["imgs_AB", "imgs_C", "seg_C"],
                source_key="imgs_AB",
            ),
            monai.transforms.Spacingd(
                keys=["imgs_AB", "imgs_C", "seg_C"],
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
                keys=["imgs_AB", "imgs_C", "seg_C"],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["imgs_AB", "imgs_C", "seg_C"],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="imgs_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="imgs_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.NormalizeIntensityd(
                keys=["imgs_AB", "imgs_C"],
                channel_wise=True
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.NormalizeIntensityd(
                keys=["imgs_AB", "imgs_C"],
                channel_wise=True
            )
        ])
    },
    "with_seg_AB": {
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
                name="imgs_AB",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=config.sequence_keys[0] + config.sequence_keys[1]
            ),
            monai.transforms.ConcatItemsd(
                keys=config.sequence_keys[2],
                name="imgs_C",
                dim=0
            ),
            monai.transforms.DeleteItemsd(keys=config.sequence_keys[2]),
            monai.transforms.ConcatItemsd(
                keys=[config.seg_keys[0], config.seg_keys[1]],
                name="seg_AB",
                dim=0
            ),
            monai.transforms.DeleteItemsd(
                keys=[config.seg_keys[0], config.seg_keys[1]]
            ),
            monai.transforms.CropForegroundd(
                keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
                source_key="imgs_AB",
            ),
            monai.transforms.Spacingd(
                keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
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
            )
        ]),
        "train_transforms": monai.transforms.Compose([
            monai.transforms.RandAffined(
                keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
                prob=1.0,
                rotate_range=0.1,
                scale_range=0.1,
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                padding_mode="zeros"
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
                label_key="seg_C",
                spatial_size=config.PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            # images_AB and images_C have different number of channels, which
            # leads to an error when processed together by RandGaussianNoised
            monai.transforms.RandGaussianNoised(
                keys="imgs_AB",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.RandGaussianNoised(
                keys="imgs_C",
                prob=1.0,
                mean=0,
                std=20
            ),
            monai.transforms.NormalizeIntensityd(
                keys=["imgs_AB", "imgs_C"],
                channel_wise=True
            )
        ]),
        "eval_transforms": monai.transforms.Compose([
            monai.transforms.NormalizeIntensityd(
                keys=["imgs_AB", "imgs_C"],
                channel_wise=True
            )
        ])
    },
    "compare": monai.transforms.Compose([
        monai.transforms.LoadImaged(
            keys=(
                config.sequence_keys[0] +
                config.sequence_keys[1] +
                config.seg_keys
            ),
            image_only=False,
            ensure_channel_first=True
        ),
        monai.transforms.ConcatItemsd(
            keys=config.sequence_keys[0] + config.sequence_keys[1],
            name="imgs_AB",
            dim=0
        ),
        monai.transforms.DeleteItemsd(
            keys=config.sequence_keys[0] + config.sequence_keys[1]
        ),
        monai.transforms.ConcatItemsd(
            keys=[config.seg_keys[0], config.seg_keys[1]],
            name="seg_AB",
            dim=0
        ),
        monai.transforms.DeleteItemsd(
            keys=[config.seg_keys[0], config.seg_keys[1]]
        ),
        monai.transforms.CropForegroundd(
            keys=["imgs_AB", "seg_AB", "seg_C"],
            source_key="imgs_AB",
        ),
        monai.transforms.Spacingd(
            keys=["imgs_AB", "seg_AB", "seg_C"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest", "nearest"),
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
        monai.transforms.NormalizeIntensityd(
            keys="imgs_AB",
            channel_wise=True
        ),
        monai.transforms.ConcatItemsd(
            keys=["imgs_AB", "seg_AB"],
            name="input_AB",
            dim=0
        ),
        monai.transforms.DeleteItemsd(keys="seg_AB")
    ])
}
