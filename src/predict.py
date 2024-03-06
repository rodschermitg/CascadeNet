import json
import os

import matplotlib
import monai
import torch

import config_base_model as config
import models
import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

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
        **config.MODEL_KWARGS_AB2C
    ).to(device)
    model.load_state_dict(checkpoint["net_AB2C_state_dict"])
    model.eval()
    model_list.append(model)

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

for batch in dataloader:
    images = batch["images_AB"].to(device)
    label = batch["label_C"].to(device)
    label = torch.argmax(label, dim=1)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            preds = [
                monai.inferers.sliding_window_inference(
                    inputs=images,
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

    images_list = [
        images[0, channel][None].cpu() for channel in range(images.shape[1])
    ]

    patient_name = utils.get_patient_name(
        batch["label_C_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        images_list + [label.cpu()] + [pred.cpu()],
        title=patient_name,
        labels=config.sequence_keys_AB + ["label", "pred"]
    )
