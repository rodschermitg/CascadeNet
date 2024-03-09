"""run from project root: python3 -m scripts.visualize_transforms"""
import json
import os

import matplotlib
import monai
import torch

from src import config_base_model as config
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

# set prob = 1.0 for all non-deterministic transforms to visualize
for transform in config.train_transforms.transforms:
    if hasattr(transform, "prob"):
        transform.prob = 1.0
transforms = monai.transforms.Compose([
    config.base_transforms,
    config.train_transforms
])

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(data["train"], transforms)
dataloader = monai.data.DataLoader(dataset, batch_size=1)

for batch in dataloader:
    imgs_AB = batch["imgs_AB"]
    imgs_C = batch["imgs_C"]
    seg = batch["seg_C"]
    seg = torch.argmax(seg, dim=1)

    imgs_AB_list = [
        imgs_AB[:, channel] for channel in range(2*config.num_sequences)
    ]
    imgs_C_list = [
        monai.visualize.utils.blend_images(imgs_C[:, channel], seg)
        for channel in range(config.num_sequences)
    ]

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        imgs_AB_list + imgs_C_list,
        title=patient_name,
        slice_dim=0,
        labels=config.sequence_keys_AB + config.sequence_keys_C
    )
