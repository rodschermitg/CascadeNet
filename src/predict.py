import json
import os

import matplotlib
import monai
import torch

import config
from probabilistic_unet.model import ProbabilisticUnet
from utils import create_slice_plots


matplotlib.use("TkAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint = torch.load(
    os.path.join(config.model_dir, f"{config.MODEL_NAME}.tar"),
    map_location=device
)
model = ProbabilisticUnet(**config.MODEL_KWARGS_A2B).to(device)
model.load_state_dict(checkpoint["net_A2B_state_dict"])
decode_onehot = monai.transforms.AsDiscrete(argmax=True, keepdim=True)

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data=data["test"],
    transform=monai.transforms.Compose([
        config.base_transforms,
        config.eval_transforms
    ])
)
print(f"Using {len(dataset)} test samples")

dataloader = monai.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

model.eval()
for batch in dataloader:
    images = batch["images_A"].to(device)
    label = batch["label"].to(device)
    label = decode_onehot(label.squeeze(0))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            pred = monai.inferers.sliding_window_inference(
                inputs=images,
                roi_size=config.PATCH_SIZE,
                sw_batch_size=config.BATCH_SIZE,
                predictor=model
            )
    pred = decode_onehot(pred.squeeze(0))

    images_list = [
        images[0, channel][None].cpu() for channel in range(images.shape[1])
    ]

    file_path = batch["label_meta_dict"]["filename_or_obj"][0]
    start_idx = file_path.find("Patient")
    end_idx = file_path.find(os.sep, start_idx)
    patient_name = file_path[start_idx:end_idx]

    create_slice_plots(
        images_list + [label.cpu()] + [pred.cpu()],
        title=patient_name,
        labels=config.modality_keys_A + ["label", "pred"]
    )
