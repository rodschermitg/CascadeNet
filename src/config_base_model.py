import os

import monai
import torch


# input path
data_dir = os.path.join("data", "processed", "patients")
train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")
raw_data_dir = os.path.join("data", "raw", "patients")
DATA_FILENAME = "dataset.json"

# random state
RANDOM_STATE = 42

# data
TEST_SIZE = 0.1
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 2
TIMESTEPS = ["A", "B", "C"]
SEQUENCES = ["CT1", "FLAIR", "T1", "T2"]
num_sequences = len(SEQUENCES)
sequence_keys_AB = (
    [f"{sequence}_{TIMESTEPS[0]}" for sequence in SEQUENCES] +
    [f"{sequence}_{TIMESTEPS[1]}" for sequence in SEQUENCES]
)
sequence_keys_C = [f"{sequence}_{TIMESTEPS[2]}" for sequence in SEQUENCES]

# transforms
base_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=sequence_keys_AB + sequence_keys_C + ["label_C"],
        image_only=False,
        ensure_channel_first=True
    ),
    monai.transforms.ConcatItemsd(
        keys=sequence_keys_AB,
        name="images_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=sequence_keys_AB),
    monai.transforms.ConcatItemsd(
        keys=sequence_keys_C,
        name="images_C",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=sequence_keys_C),
    monai.transforms.CropForegroundd(
        keys=["images_AB", "images_C", "label_C"],
        source_key="images_AB",
    ),
    monai.transforms.Spacingd(
        keys=["images_AB", "images_C", "label_C"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "bilinear", "nearest"),
    ),
    monai.transforms.ThresholdIntensityd(
        keys="label_C",
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="label_C", to_onehot=NUM_CLASSES),
    # monai.transforms.Orientationd(
    #     keys=["images_AB", "images_C", "label_C"],
    #     axcodes="SPL",
    # ),
])
train_transforms = monai.transforms.Compose([
    monai.transforms.RandAffined(
        keys=["images_AB", "images_C", "label_C"],
        prob=1.0,
        rotate_range=0.1,
        scale_range=0.1,
        mode=("bilinear", "bilinear", "nearest"),
        padding_mode="zeros"
    ),
    monai.transforms.RandCropByPosNegLabeld(
        keys=["images_AB", "images_C", "label_C"],
        label_key="label_C",
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=1,
    ),
    # images_AB and images_C have different number of channels, which leads to
    # an error when processed together by RandGaussianNoised
    monai.transforms.RandGaussianNoised(
        keys="images_AB",
        prob=1.0,
        mean=0,
        std=20
    ),
    monai.transforms.RandGaussianNoised(
        keys="images_C",
        prob=1.0,
        mean=0,
        std=20
    ),
    monai.transforms.NormalizeIntensityd(
        keys=["images_AB", "images_C"],
        channel_wise=True
    )
])
eval_transforms = monai.transforms.Compose([
    monai.transforms.NormalizeIntensityd(
        keys=["images_AB", "images_C"],
        channel_wise=True
    )
])

# model
MODEL_KWARGS_AB2C = {
    "in_channels": 2 * num_sequences,
    "out_channels": NUM_CLASSES,
    "latent_size": 3,
    "temperature": 0.28,
    "task_kwargs": {
        "activation_kwargs": {"inplace": True}
    },
    "prior_kwargs": {
        "encoder_kwargs": {"norm_depth": "full"},
        "n_components": 9
    },
    "posterior_kwargs": {
        "encoder_kwargs": {"norm_depth": "full"},
        "n_components": 9
    }
}
MODEL_KWARGS_C2AB = {
    "in_channels": num_sequences + NUM_CLASSES,
    "out_channels": 2 * num_sequences,
    "latent_size": 3,
    "temperature": 0.28,
    "task_kwargs": {
        "output_activation_op": torch.nn.Tanh,
        "activation_kwargs": {"inplace": True}
    },
    "prior_kwargs": {
        "encoder_kwargs": {"norm_depth": "full"},
        "n_components": 9
    },
    "posterior_kwargs": {
        "encoder_kwargs": {"norm_depth": "full"},
        "n_components": 9
    }
}

# train
LR = 1e-4
WEIGHT_DECAY = 1e-5
FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 1
VAL_INTERVAL = 5
DISPLAY_INTERVAL = 5
CYCLE_WEIGHT = 1
KL_WEIGHT = 1
SAVE_MODEL_EACH_FOLD = True

# output path
checkpoint_dir = os.path.join("checkpoints", "base_model")
MODEL_NAME = "cascade_net"
logs_dir = os.path.join("logs", "base_model")
train_logs_path = os.path.join(logs_dir, "train_logs.json")
pred_loss_plot_path = os.path.join(logs_dir, "pred_loss_plot.png")
cycle_loss_plot_path = os.path.join(logs_dir, "cycle_loss_plot.png")
kl_loss_plot_path = os.path.join(logs_dir, "kl_loss_plot.png")
metric_plot_path = os.path.join(logs_dir, "metric_plot.png")
cv_fold_logs_path = os.path.join(logs_dir, "cv_fold_logs.json")
test_logs_path = os.path.join(logs_dir, "test_logs.json")
