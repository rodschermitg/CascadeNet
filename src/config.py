import os

import monai
import torch


# input path
data_dir = os.path.join("data", "processed", "patients")
train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")
raw_data_dir = os.path.join("data", "raw", "patients")
DATA_FILENAME = "dataset.json"

# model
MODEL_KWARGS_A2B = {
    "input_channels": 8,
    "label_channels": 2,
    "num_classes": 2,
    "num_filters": [32, 64, 128, 512],
    "activation_layer": torch.nn.LogSoftmax(dim=1),
    "latent_dim": 2,
    "full_cov": True,
    "n_components": 9,
    "temperature": 0.28
 }
MODEL_KWARGS_B2A = {
    "input_channels": 6,
    "label_channels": 8,
    "num_classes": 8,
    "num_filters": [32, 64, 128, 512],
    "activation_layer": torch.nn.Tanh(),
    "latent_dim": 2,
    "full_cov": True,
    "n_components": 9,
    "temperature": 0.28
}

# random state
RANDOM_STATE = 42

# data
TEST_SIZE = 0.1
MODALITIES = ["CT1", "FLAIR", "T1", "T2"]
modality_keys_A = (
    [modality + "_1" for modality in MODALITIES] +
    [modality + "_2" for modality in MODALITIES]
)
modality_keys_B = [modality + "_3" for modality in MODALITIES]
PATCH_SIZE = (96, 96, 96)

# transforms
base_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=modality_keys_A + modality_keys_B + ["label"],
        image_only=False,
        ensure_channel_first=True
    ),
    monai.transforms.ConcatItemsd(
        keys=modality_keys_A,
        name="images_A",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=modality_keys_A),
    monai.transforms.ConcatItemsd(
        keys=modality_keys_B,
        name="images_B",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=modality_keys_B),
    monai.transforms.CropForegroundd(
        keys=["images_A", "images_B", "label"],
        source_key="images_A",
    ),
    monai.transforms.ThresholdIntensityd(
        keys="label",
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="label", to_onehot=2),
    # monai.transforms.Orientationd(
    #     keys=["images_A", "images_B", "label"],
    #     axcodes="SPL",
    # ),
])
train_transforms = monai.transforms.Compose([
    monai.transforms.RandAffined(
        keys=["images_A", "images_B", "label"],
        prob=0.1,
        rotate_range=0.1,
        scale_range=0.1,
        mode=("bilinear", "bilinear", "nearest")
    ),
    monai.transforms.RandCropByPosNegLabeld(
        keys=["images_A", "images_B", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=1,
    ),
    # images_A and images_B have different number of channels, which leads to
    # an error when processed together by RandGaussianNoised
    monai.transforms.RandGaussianNoised(
        keys=["images_A"],
        prob=0.1,
        mean=0.0,
        std=0.1
    ),
    monai.transforms.RandGaussianNoised(
        keys=["images_B"],
        prob=0.1,
        mean=0.0,
        std=0.1
    ),
    monai.transforms.NormalizeIntensityd(
        keys=["images_A", "images_B"],
        channel_wise=True
    )
])
eval_transforms = monai.transforms.Compose([
    monai.transforms.NormalizeIntensityd(
        keys=["images_A", "images_B"],
        channel_wise=True
    )
])

# train
LR = 1e-4
FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 1
VAL_INTERVAL = 5
DISPLAY_INTERVAL = 5
CYCLE_WEIGHT = 1
KL_WEIGHT = 1
SAVE_MODEL_EACH_FOLD = True

# output path
model_dir = os.path.join("models")
MODEL_NAME = "probunet_patients"
train_logs_path = os.path.join("logs", "train_logs.json")
pred_loss_plot_path = os.path.join("logs", "pred_loss_plot.png")
cycle_loss_plot_path = os.path.join("logs", "cycle_loss_plot.png")
kl_loss_plot_path = os.path.join("logs", "kl_loss_plot.png")
val_metric_plot_path = os.path.join("logs", "val_metric_plot.png")
