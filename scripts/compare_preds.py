"""run from project root: python3 -m scripts.compare_preds"""
import json
import os

import matplotlib
import monai
import torch

from src import config
from src import models
from src import transforms
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

model_dict = {task_key: [] for task_key in config.INPUT_DICT.keys()}
for task_key in model_dict.keys():
    for fold in range(config.FOLDS):
        checkpoint_path = os.path.join(
            "checkpoints",
            task_key,
            f"{config.MODEL_NAME}_fold{fold}.tar"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = models.ProbabilisticSegmentationNet(
            **config.NET_AB2C_KWARGS_DICT[task_key]
        ).to(device)
        model.load_state_dict(checkpoint["net_AB2C_state_dict"])
        model.eval()
        model_dict[task_key].append(model)

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data["test"],
    transforms.transforms_dict["compare"]
)
dataloader = monai.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)
print(f"Using {len(dataset)} test samples")

pred_dict = {task_key: None for task_key in model_dict.keys()}

for batch in dataloader:
    seg = batch["seg_C"].to(device)
    seg = torch.argmax(seg, dim=1)

    for task_key in model_dict.keys():
        input = batch[config.INPUT_DICT[task_key]].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                preds = [
                    monai.inferers.sliding_window_inference(
                        input,
                        roi_size=config.PATCH_SIZE,
                        sw_batch_size=config.BATCH_SIZE,
                        predictor=model
                    )
                    for model in model_dict[task_key]
                ]
        preds = [torch.nn.functional.softmax(pred, dim=1) for pred in preds]
        preds = torch.cat(preds, dim=0)
        pred = torch.mean(preds, dim=0, keepdim=True)
        pred = torch.argmax(pred, dim=1)
        pred_dict[task_key] = pred

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        [tensor.cpu() for tensor in [seg] + list(pred_dict.values())],
        title=patient_name,
        labels=["seg"] + list(pred_dict.keys())
    )
