"""run from project root: python3 -m scripts.plot_cascade_net_output"""
import json
import os

import matplotlib
import monai
import torch

from src import config
from src import models
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint_list = [
    torch.load(
        os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_fold{fold}.tar"),
        map_location=device
    )
    for fold in range(config.FOLDS)
]
net_AB2C_list = [
    models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS_AB2C).to(device)
    for _ in range(config.FOLDS)
]
net_C2AB_list = [
    models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS_C2AB).to(device)
    for _ in range(config.FOLDS)
]
for net_AB2C, checkpoint in zip(net_AB2C_list, checkpoint_list):
    net_AB2C.load_state_dict(checkpoint["net_AB2C_state_dict"])
    net_AB2C.eval()
for net_C2AB, checkpoint in zip(net_C2AB_list, checkpoint_list):
    net_C2AB.load_state_dict(checkpoint["net_C2AB_state_dict"])
    net_C2AB.eval()

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data=data["train"],
    transform=monai.transforms.Compose([
        config.base_transforms,
        config.eval_transforms
    ])
)
print(f"Using {len(dataset)} training samples")

dataloader = monai.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

for batch in dataloader:
    real_AB = batch["images_AB"].to(device)
    real_C = batch["images_C"].to(device)
    label_C = batch["label_C"]
    label_C = torch.argmax(label_C, dim=1)  # decode one-hot labels

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # net_AB2C
            preds_C = [
                monai.inferers.sliding_window_inference(
                    inputs=real_AB,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=net_AB2C
                )
                for net_AB2C in net_AB2C_list
            ]
            preds_C = [
                torch.nn.functional.softmax(pred, dim=1)
                for pred in preds_C
            ]
            preds_C = torch.cat(preds_C, dim=0)
            pred_C = torch.mean(preds_C, dim=0, keepdim=True)

            # net_C2AB
            mean = pred_C.mean(dim=(2, 3, 4), keepdim=True)
            std = pred_C.std(dim=(2, 3, 4), keepdim=True)

            recs_AB = [
                monai.inferers.sliding_window_inference(
                    inputs=torch.cat(((pred_C-mean)/std, real_C), dim=1),
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=net_C2AB
                )
                for net_C2AB in net_C2AB_list
            ]
            recs_AB = [
                torch.nn.functional.softmax(rec, dim=1)
                for rec in recs_AB
            ]
            recs_AB = torch.cat(recs_AB, dim=0)
            rec_AB = torch.mean(recs_AB, dim=0, keepdim=True)

    rec_AB_list = [
        rec_AB[0, channel][None].cpu() for channel in range(rec_AB.shape[1])
    ]
    pred_C_list = [
        pred_C[0, channel][None].cpu() for channel in range(pred_C.shape[1])
    ]

    patient_name = utils.get_patient_name(
        batch["label_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        rec_AB_list + pred_C_list + [label_C],
        title=patient_name,
        labels=config.sequence_keys_AB + ["pred[0]", "pred[1]", "label"]
    )
