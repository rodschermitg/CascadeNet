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

base_transforms = transforms.transforms_dict[config.TASK]["base_transforms"]
train_transforms = transforms.transforms_dict[config.TASK]["train_transforms"]
# set prob = 1.0 for all non-deterministic transforms to visualize
for transform in train_transforms.transforms:
    if hasattr(transform, "prob"):
        transform.prob = 1.0

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data["test"],
    monai.transforms.Compose([
        *base_transforms.transforms,
        *train_transforms.transforms
    ])
)
dataloader = monai.data.DataLoader(dataset, batch_size=1)

for batch in dataloader:
    input_AB = batch[config.INPUT_DICT_AB[config.TASK]]
    input_C = batch[config.INPUT_DICT_C[config.TASK]]
    seg_C = batch["seg_C"]
    seg_C = torch.argmax(seg_C, dim=1)

    input_AB_list = [
        input_AB[:, channel_idx]
        for channel_idx in range(input_AB.shape[1])
    ]
    input_C_list = [
        input_C[:, channel_idx]
        for channel_idx in range(input_C.shape[1])
    ]

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        input_AB_list + input_C_list + [seg_C],
        title=patient_name,
        slice_dim=0
    )
