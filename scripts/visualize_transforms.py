"""run from project root: python3 -m scripts.visualize_transforms"""
import json
import os

import matplotlib
import monai

from src import config
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=config.modality_keys_A + config.modality_keys_B + ["label"],
        # image_only=True,
        image_only=False,
        ensure_channel_first=True
    ),
    monai.transforms.ConcatItemsd(
        keys=config.modality_keys_A,
        name="images_A",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=config.modality_keys_A),
    monai.transforms.ConcatItemsd(
        keys=config.modality_keys_B,
        name="images_B",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=config.modality_keys_B),
    monai.transforms.CropForegroundd(
        keys=["images_A", "images_B", "label"],
        source_key="images_A",
    ),
    monai.transforms.ThresholdIntensityd(
        keys="label",
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="label", to_onehot=2),
    # monai.transforms.Orientationd(
    #     keys=["images_A", "images_B", "label"],
    #     axcodes="SPL",
    # ),
    monai.transforms.RandAffined(
        keys=["images_A", "images_B", "label"],
        # prob=0.1,
        prob=1.0,
        rotate_range=0.1,
        scale_range=0.1,
        mode=("bilinear", "bilinear", "nearest")
    ),
    monai.transforms.RandCropByPosNegLabeld(
        keys=["images_A", "images_B", "label"],
        label_key="label",
        spatial_size=config.PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=1,
    ),
    # images_A and images_B have different number of channels, which leads to
    # an error when processed together by RandGaussianNoised
    monai.transforms.RandGaussianNoised(
        keys=["images_A"],
        # prob=0.1,
        prob=1.0,
        mean=0.0,
        std=0.1
    ),
    monai.transforms.RandGaussianNoised(
        keys=["images_B"],
        # prob=0.1,
        prob=1.0,
        mean=0.0,
        std=0.1
    ),
    monai.transforms.NormalizeIntensityd(
        keys=["images_A", "images_B"],
        channel_wise=True
    )
])

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(data=data["train"], transform=transforms)
dataloader = monai.data.DataLoader(dataset, batch_size=1)
decode_onehot = monai.transforms.AsDiscrete(argmax=True, keepdim=True)

for batch in dataloader:
    images_A = batch["images_A"].squeeze(0)
    images_B = batch["images_B"].squeeze(0)
    label_B = decode_onehot(batch["label"].squeeze(0))

    images_A_list = [
        images_A[channel][None] for channel in range(images_A.shape[0])
    ]
    images_B_list = [
        monai.visualize.utils.blend_images(images_B[channel][None], label_B)
        for channel in range(images_B.shape[0])
    ]

    patient_name = utils.get_patient_name(
        batch["label_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        images_A_list + images_B_list,
        title=patient_name,
        slice_dim=0,
        labels=config.modality_keys_A + config.modality_keys_B
    )
