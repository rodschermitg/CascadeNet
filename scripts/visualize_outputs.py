"""run from project root: python3 -m scripts.visualize_outputs"""
import json
import os

import matplotlib
import monai
import torch

from probabilistic_unet.model import ProbabilisticUnet
from src import config
from src import utils


matplotlib.use("TkAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint = torch.load(
    os.path.join(config.model_dir, f"{config.MODEL_NAME}.tar"),
    map_location=device
)
net_A2B = ProbabilisticUnet(**config.MODEL_KWARGS_A2B).to(device)
net_A2B.load_state_dict(checkpoint["net_A2B_state_dict"])
net_B2A = ProbabilisticUnet(**config.MODEL_KWARGS_B2A).to(device)
net_B2A.load_state_dict(checkpoint["net_B2A_state_dict"])

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

net_A2B.eval()
net_B2A.eval()

for batch in dataloader:
    real_A = batch["images_A"].to(device)
    real_B = batch["images_B"].to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            pred_B = monai.inferers.sliding_window_inference(
                inputs=real_A,
                roi_size=config.PATCH_SIZE,
                sw_batch_size=config.BATCH_SIZE,
                predictor=net_A2B
            )

            mean = pred_B.mean(dim=(2, 3, 4), keepdim=True)
            std = pred_B.std(dim=(2, 3, 4), keepdim=True)
            rec_A = monai.inferers.sliding_window_inference(
                inputs=torch.cat(((pred_B-mean)/std, real_B), dim=1),
                roi_size=config.PATCH_SIZE,
                sw_batch_size=config.BATCH_SIZE,
                predictor=net_B2A
            )

    rec_A_list = [
        rec_A[0, channel][None].cpu() for channel in range(rec_A.shape[1])
    ]
    pred_B_list = [
        pred_B[0, channel][None].cpu() for channel in range(pred_B.shape[1])
    ]

    patient_name = utils.get_patient_name(
        batch["label_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        rec_A_list + pred_B_list,
        title=patient_name,
        labels=config.modality_keys_A + ["label_0", "label_1"]
    )
