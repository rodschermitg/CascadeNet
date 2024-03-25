"""run from project root: python3 -m scripts.visualize_transforms"""
import json
import os

import matplotlib
import monai
import torch

from src import config
from src import transforms
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

base_transforms = transforms.transforms_dict["base_model"]["base_transforms"]
train_transforms = transforms.transforms_dict["base_model"]["train_transforms"]
# set prob = 1.0 for all non-deterministic transforms to visualize
for transform in train_transforms.transforms:
    if hasattr(transform, "prob"):
        transform.prob = 1.0

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data["train"],
    monai.transforms.Compose([
        *base_transforms.transforms,
        *train_transforms.transforms
    ])
)
dataloader = monai.data.DataLoader(dataset, batch_size=1)

for batch in dataloader:
    img_AB = batch["img_AB"]
    img_C = batch["img_C"]
    seg = batch["seg_C"]
    seg = torch.argmax(seg, dim=1)

    img_AB_list = [
        img_AB[:, channel_idx]
        for channel_idx in range(config.NUM_INPUTS*config.num_sequences)
    ]
    img_C_list = [
        monai.visualize.utils.blend_images(img_C[:, channel_idx], seg)
        for channel_idx in range(config.num_sequences)
    ]

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        img_AB_list + img_C_list,
        title=patient_name,
        slice_dim=0,
        labels=(
            config.sequence_keys[0] +
            config.sequence_keys[1] +
            config.sequence_keys[2] +
            ["seg_C"]
        )
    )
