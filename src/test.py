import json
import os

import monai
import torch

import config
import models
import transforms
import utils


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

discretize = monai.transforms.AsDiscrete(
    argmax=True,
    to_onehot=config.NUM_CLASSES
)

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
print(f"Using {len(dataset)} test samples\n")

dataloader = monai.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    metric_name=("precision", "recall"),
    include_background=False
)
precision_list = []
recall_list = []

test_logs = {
    "individual": {},
    "mean": {},
    "std": {}
}

for batch in dataloader:
    input = batch[config.INPUT_DICT_AB[config.TASK]].to(device)
    seg = batch["seg_C"].to(device)

    patient_name = utils.get_patient_name(
        batch["seg_C_meta_dict"]["filename_or_obj"][0]
    )

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

    # store discretized batches in lists for metric functions
    pred = [discretize(p) for p in monai.data.decollate_batch(pred)]
    seg = monai.data.decollate_batch(seg)

    # metric results are stored internally
    dice_fn(pred, seg)
    confusion_matrix_fn(pred, seg)

    # store precision and recall in separate lists for later calculations
    precision_list.append(confusion_matrix_fn.aggregate()[0].item())
    recall_list.append(confusion_matrix_fn.aggregate()[1].item())
    confusion_matrix_fn.reset()

    if patient_name not in test_logs["individual"]:
        test_logs["individual"][patient_name] = {
            "dice": [], "precision": [], "recall": []
        }

    test_logs["individual"][patient_name]["dice"].append(
        dice_fn.get_buffer()[-1].item()
    )
    test_logs["individual"][patient_name]["precision"].append(
        precision_list[-1]
    )
    test_logs["individual"][patient_name]["recall"].append(recall_list[-1])

    num_scores = len(test_logs['individual'][patient_name]['dice'])
    print(f"{patient_name} ({num_scores}):")
    print(f"\tdice: {dice_fn.get_buffer()[-1].item():.4f}")
    print(f"\tprecision: {precision_list[-1]:.4f}")
    print(f"\trecall: {recall_list[-1]:.4f}")

mean_dice = torch.mean(dice_fn.get_buffer()).item()
mean_precision = torch.mean(torch.tensor(precision_list)).item()
mean_recall = torch.mean(torch.tensor(recall_list)).item()
std_dice = torch.std(dice_fn.get_buffer(), correction=0).item()
std_precision = torch.std(torch.tensor(precision_list), correction=0).item()
std_recall = torch.std(torch.tensor(recall_list), correction=0).item()

test_logs["mean"]["dice"] = mean_dice
test_logs["mean"]["precision"] = mean_precision
test_logs["mean"]["recall"] = mean_recall
test_logs["std"]["dice"] = std_dice
test_logs["std"]["precision"] = std_precision
test_logs["std"]["recall"] = std_recall

print(f"\nmean dice: {mean_dice:.4f}, std dice: {std_dice:.4f}")
print(
    f"mean precision: {mean_precision:.4f}, std precision: {std_precision:.4f}"
)
print(f"mean recall: {mean_recall:.4f}, std recall: {std_recall:.4f}")

with open(config.test_logs_path, "w") as test_logs_file:
    json.dump(test_logs, test_logs_file, indent=4)
