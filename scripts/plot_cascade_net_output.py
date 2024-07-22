"""run from project root: python3 -m scripts.plot_cascade_net_output"""
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

net_AB2C_list = []
net_C2AB_list = []
for fold in range(config.FOLDS):
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{config.MODEL_NAME}_fold{fold}.tar"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_AB2C = models.ProbabilisticSegmentationNet(
        **config.NET_AB2C_KWARGS_DICT[config.TASK]
    ).to(device)
    net_C2AB = models.ProbabilisticSegmentationNet(
        **config.NET_C2AB_KWARGS_DICT[config.TASK]
    ).to(device)
    net_AB2C.load_state_dict(checkpoint["net_AB2C_state_dict"])
    net_C2AB.load_state_dict(checkpoint["net_C2AB_state_dict"])
    net_AB2C.eval()
    net_C2AB.eval()
    net_AB2C_list.append(net_AB2C)
    net_C2AB_list.append(net_C2AB)

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data["test"],
    monai.transforms.Compose([
        *transforms.transforms_dict[config.TASK]["base_transforms"].transforms,
        *transforms.transforms_dict[config.TASK]["eval_transforms"].transforms
    ])
)
print(f"Using {len(dataset)} training samples")

dataloader = monai.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

for batch in dataloader:
    input_AB = batch[config.INPUT_DICT_AB[config.TASK]].to(device)
    input_C = batch[config.INPUT_DICT_C[config.TASK]].to(device)
    seg_C = batch["seg_C"]
    seg_C = torch.argmax(seg_C, dim=1)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # net_AB2C
            preds_C = [
                monai.inferers.sliding_window_inference(
                    input_AB,
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

            recs_AB = [
                monai.inferers.sliding_window_inference(
                    torch.cat((pred_C, input_C), dim=1),
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
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        rec_AB_list + pred_C_list + [seg_C],
        title=patient_name,
        labels=(
            config.sequence_keys[0] +
            config.sequence_keys[1] +
            ["pred[0]", "pred[1]", "seg"]
        )
    )
