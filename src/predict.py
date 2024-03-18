import json
import os

import matplotlib
import monai
import torch

import config
import models
import transforms
import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)
print(f"Current task: {config.TASK}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

model_list = []
for fold in range(config.FOLDS):
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{config.MODEL_NAME}_fold{fold}.tar"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.ProbabilisticSegmentationNet(
        **config.NET_AB2C_KWARGS_DICT[config.TASK]
    ).to(device)
    model.load_state_dict(checkpoint["net_AB2C_state_dict"])
    model.eval()
    model_list.append(model)

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data["test"],
    monai.transforms.Compose([
        transforms.transforms_dict[config.TASK]["base_transforms"],
        transforms.transforms_dict[config.TASK]["eval_transforms"]
    ])
)
print(f"Using {len(dataset)} test samples")

dataloader = monai.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

for batch in dataloader:
    imgs = batch["imgs_AB"].to(device)
    seg = batch["seg_C"].to(device)
    seg = torch.argmax(seg, dim=1)

    if config.TASK == "with_seg_AB":
        input = torch.cat((imgs, batch["seg_AB"].to(device)), dim=1)
    else:
        input = imgs

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            preds = [
                monai.inferers.sliding_window_inference(
                    input,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=model
                )
                for model in model_list
            ]
    preds = [torch.nn.functional.softmax(pred, dim=1) for pred in preds]
    preds = torch.cat(preds, dim=0)
    pred = torch.mean(preds, dim=0, keepdim=True)
    pred = torch.argmax(pred, dim=1)

    imgs_list = [
        imgs[0, channel][None].cpu() for channel in range(imgs.shape[1])
    ]

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        imgs_list + [seg.cpu()] + [pred.cpu()],
        title=patient_name,
        labels=(
            config.sequence_keys[0] +
            config.sequence_keys[1] +
            ["label", "pred"]
        )
    )
