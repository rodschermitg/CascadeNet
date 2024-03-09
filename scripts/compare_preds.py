"""run from project root: python3 -m scripts.compare_preds"""
import importlib
import json
import os

import matplotlib
import monai
import torch

from src import config_compare
from src import models
from src import utils


config_dict = {
    model_key: importlib.import_module(f"src.{config_name}")
    for model_key, config_name in config_compare.CONFIG_NAME_DICT.items()
}

matplotlib.use("TkAgg")
monai.utils.set_determinism(config_compare.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

model_dict = {model_key: [] for model_key in config_dict.keys()}
for model_key in model_dict.keys():
    for fold in range(config_dict[model_key].FOLDS):
        checkpoint_path = os.path.join(
            config_dict[model_key].checkpoint_dir,
            f"{config_dict[model_key].MODEL_NAME}_fold{fold}.tar"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = models.ProbabilisticSegmentationNet(
            **config_dict[model_key].MODEL_KWARGS_AB2C
        ).to(device)
        model.load_state_dict(checkpoint["net_AB2C_state_dict"])
        model.eval()
        model_dict[model_key].append(model)

data_path = os.path.join(config_compare.data_dir, config_compare.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(data["test"], config_compare.transforms)
dataloader = monai.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)
print(f"Using {len(dataset)} test samples")

imgs_dict = {model_key: None for model_key in model_dict.keys()}
pred_dict = {model_key: None for model_key in model_dict.keys()}

for batch in dataloader:
    seg = batch["seg_C"].to(device)
    seg = torch.argmax(seg, dim=1)

    for model_key in model_dict.keys():
        imgs = batch[config_compare.IMGS_KEY_DICT[model_key]].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                preds = [
                    monai.inferers.sliding_window_inference(
                        imgs,
                        roi_size=config_dict[model_key].PATCH_SIZE,
                        sw_batch_size=config_dict[model_key].BATCH_SIZE,
                        predictor=model
                    )
                    for model in model_dict[model_key]
                ]
        preds = [torch.nn.functional.softmax(pred, dim=1) for pred in preds]
        preds = torch.cat(preds, dim=0)
        pred = torch.mean(preds, dim=0, keepdim=True)
        pred = torch.argmax(pred, dim=1)
        pred_dict[model_key] = pred

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        [tensor.cpu() for tensor in [seg] + list(pred_dict.values())],
        title=patient_name,
        labels=["seg"] + list(pred_dict.keys())
    )
