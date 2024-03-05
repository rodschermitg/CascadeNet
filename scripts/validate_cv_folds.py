"""run from project root: python3 -m scripts.validate_cv_folds"""
import json
import os

import monai
import torch

from src import config
from src import models


monai.utils.set_determinism(config.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint_path_list = [
    os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_fold{fold}.tar")
    for fold in range(config.FOLDS)
]
checkpoint_list = [
    torch.load(checkpoint_path, map_location=device)
    for checkpoint_path in checkpoint_path_list
]
model_list = [
    models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS_AB2C).to(device)
    for _ in range(config.FOLDS)
]
for model, checkpoint in zip(model_list, checkpoint_list):
    model.load_state_dict(checkpoint["net_AB2C_state_dict"])
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
train_precision_list = []
train_recall_list = []
val_precision_list = []
val_recall_list = []

cv_fold_logs = {
    "mean_train_dice": {f"fold{i}": None for i in range(config.FOLDS)},
    "mean_train_precision": {f"fold{i}": None for i in range(config.FOLDS)},
    "mean_train_recall": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_train_dice": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_train_precision": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_train_recall": {f"fold{i}": None for i in range(config.FOLDS)},
    "mean_val_dice": {f"fold{i}": None for i in range(config.FOLDS)},
    "mean_val_precision": {f"fold{i}": None for i in range(config.FOLDS)},
    "mean_val_recall": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_val_dice": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_val_precision": {f"fold{i}": None for i in range(config.FOLDS)},
    "std_val_recall": {f"fold{i}": None for i in range(config.FOLDS)}
}

for fold in range(config.FOLDS):
    train_indices = train_logs["fold_indices"][f"fold{fold}"]["train_indices"]
    train_data = torch.utils.data.Subset(dataset, train_indices)
    train_dataloader = monai.data.DataLoader(
        dataset=train_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for train_batch in train_dataloader:
        train_images = train_batch["images_AB"].to(device)
        train_label = train_batch["label_C"].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                train_pred = monai.inferers.sliding_window_inference(
                    inputs=train_images,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=model_list[fold]
                )
        train_pred = torch.nn.functional.softmax(train_pred, dim=1)
        # store discretized batches in lists for metric functions
        train_pred = [
            discretize(pred) for pred in monai.data.decollate_batch(train_pred)
        ]
        train_label = monai.data.decollate_batch(train_label)

        # metric results are stored internally
        dice_fn(train_pred, train_label)
        confusion_matrix_fn(train_pred, train_label)

        # store precision and recall in separate lists for later calculations
        train_precision_list.append(confusion_matrix_fn.aggregate()[0].item())
        train_recall_list.append(confusion_matrix_fn.aggregate()[1].item())
        confusion_matrix_fn.reset()

    mean_train_dice = torch.mean(dice_fn.get_buffer()).item()
    mean_train_precision = torch.mean(
        torch.tensor(train_precision_list)
    ).item()
    mean_train_recall = torch.mean(torch.tensor(train_recall_list)).item()
    std_train_dice = torch.std(dice_fn.get_buffer(), correction=0).item()
    std_train_precision = torch.std(
        torch.tensor(train_precision_list),
        correction=0
    ).item()
    std_train_recall = torch.std(
        torch.tensor(train_recall_list),
        correction=0
    ).item()

    dice_fn.reset()

    cv_fold_logs["mean_train_dice"][f"fold{fold}"] = mean_train_dice
    cv_fold_logs["mean_train_precision"][f"fold{fold}"] = mean_train_precision
    cv_fold_logs["mean_train_recall"][f"fold{fold}"] = mean_train_recall
    cv_fold_logs["std_train_dice"][f"fold{fold}"] = std_train_dice
    cv_fold_logs["std_train_precision"][f"fold{fold}"] = std_train_precision
    cv_fold_logs["std_train_recall"][f"fold{fold}"] = std_train_recall

    print(f"\nfold {fold}:")
    print(
        f"\tmean train dice: {mean_train_dice:.4f}, "
        f"std train dice: {std_train_dice:.4f}"
    )
    print(
        f"\tmean train precision: {mean_train_precision:.4f}, "
        f"std train precision: {std_train_precision:.4f}"
    )
    print(
        f"\tmean train recall: {mean_train_recall:.4f}, "
        f"std train recall: {std_train_recall:.4f}"
    )

    val_indices = train_logs["fold_indices"][f"fold{fold}"]["val_indices"]
    val_data = torch.utils.data.Subset(dataset, val_indices)
    val_dataloader = monai.data.DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for val_batch in val_dataloader:
        val_images = val_batch["images_AB"].to(device)
        val_label = val_batch["label_C"].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                val_pred = monai.inferers.sliding_window_inference(
                    inputs=val_images,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=model_list[fold]
                )
        val_pred = torch.nn.functional.softmax(val_pred, dim=1)
        val_pred = [
            discretize(pred) for pred in monai.data.decollate_batch(val_pred)
        ]
        val_label = monai.data.decollate_batch(val_label)

        dice_fn(val_pred, val_label)
        confusion_matrix_fn(val_pred, val_label)

        val_precision_list.append(confusion_matrix_fn.aggregate()[0].item())
        val_recall_list.append(confusion_matrix_fn.aggregate()[1].item())
        confusion_matrix_fn.reset()

    mean_val_dice = torch.mean(dice_fn.get_buffer()).item()
    mean_val_precision = torch.mean(torch.tensor(val_precision_list)).item()
    mean_val_recall = torch.mean(torch.tensor(val_recall_list)).item()
    std_val_dice = torch.std(dice_fn.get_buffer(), correction=0).item()
    std_val_precision = torch.std(
        torch.tensor(val_precision_list),
        correction=0
    ).item()
    std_val_recall = torch.std(
        torch.tensor(val_recall_list),
        correction=0
    ).item()

    dice_fn.reset()

    cv_fold_logs["mean_val_dice"][f"fold{fold}"] = mean_val_dice
    cv_fold_logs["mean_val_precision"][f"fold{fold}"] = mean_val_precision
    cv_fold_logs["mean_val_recall"][f"fold{fold}"] = mean_val_recall
    cv_fold_logs["std_val_dice"][f"fold{fold}"] = std_val_dice
    cv_fold_logs["std_val_precision"][f"fold{fold}"] = std_val_precision
    cv_fold_logs["std_val_recall"][f"fold{fold}"] = std_val_recall

    print(
        f"\tmean val dice: {mean_val_dice:.4f}, "
        f"std val dice: {std_val_dice:.4f}"
    )
    print(
        f"\tmean val precision: {mean_val_precision:.4f}, "
        f"std val precision: {std_val_precision:.4f}"
    )
    print(
        f"\tmean val recall: {mean_val_recall:.4f}, "
        f"std val recall: {std_val_recall:.4f}"
    )

with open(config.cv_fold_logs_path, "w") as cv_fold_logs_file:
    json.dump(cv_fold_logs, cv_fold_logs_file, indent=4)
