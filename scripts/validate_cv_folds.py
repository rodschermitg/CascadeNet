"""run from project root: python3 -m scripts.validate_cv_folds"""
import json
import os

import monai
import torch

from probabilistic_unet.model import ProbabilisticUnet
from src import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint_list = [
    torch.load(
        os.path.join(config.model_dir, f"{config.MODEL_NAME}_fold{fold}.tar"),
        map_location=device
    )
    for fold in range(config.FOLDS)
]
model_list = [
    ProbabilisticUnet(**config.MODEL_KWARGS_A2B).to(device)
    for _ in range(config.FOLDS)
]
for model, checkpoint in zip(model_list, checkpoint_list):
    model.load_state_dict(checkpoint["net_A2B_state_dict"])
    model.eval()

discretize = monai.transforms.AsDiscrete(
    argmax=True,
    to_onehot=config.NUM_CLASSES
)

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.CacheDataset(
    data=data["train"],
    transform=monai.transforms.Compose([
        config.base_transforms,
        config.eval_transforms
    ]),
    num_workers=num_workers
)

with open(config.train_logs_path, "r") as train_logs_file:
    train_logs = json.load(train_logs_file)

dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    metric_name=("precision", "recall"),
    include_background=False
)
precision_list = []
recall_list = []

for fold in range(config.FOLDS):
    val_indices = train_logs["fold_indices"][f"fold{fold}"]["val_indices"]
    val_data = torch.utils.data.Subset(dataset, val_indices)
    dataloader = monai.data.DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for batch in dataloader:
        images = batch["images"].to(device)
        label = batch["label"].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                preds = monai.inferers.sliding_window_inference(
                    inputs=images,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=model_list[fold]
                )

        # store discretized batches in lists for metric functions
        preds = [
            discretize(pred) for pred in monai.data.decollate_batch(preds)
        ]
        label = monai.data.decollate_batch(label)

        # metric results are stored internally
        dice_fn(preds, label)
        confusion_matrix_fn(preds, label)

        # store precision and recall in separate lists for later calculations
        precision_list.append(confusion_matrix_fn.aggregate()[0].item())
        recall_list.append(confusion_matrix_fn.aggregate()[1].item())
        confusion_matrix_fn.reset()

    mean_dice = torch.mean(dice_fn.get_buffer()).item()
    std_dice = torch.std(dice_fn.get_buffer(), correction=0).item()
    mean_precision = torch.mean(torch.tensor(precision_list))
    std_precision = torch.std(torch.tensor(precision_list), correction=0)
    mean_recall = torch.mean(torch.tensor(recall_list))
    std_recall = torch.std(torch.tensor(recall_list), correction=0)

    print(f"\nfold {fold}:")
    print(f"\tmean dice: {mean_dice:.4f}, std dice: {std_dice:.4f}")
    print(
        f"\tmean precision: {mean_precision:.4f}, "
        f"std precision: {std_precision:.4f}"
    )
    print(f"\tmean recall: {mean_recall:.4f}, std recall: {std_recall:.4f}")
