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
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 2
TIMESTEPS = ["A", "B", "C"]
NUM_INPUT_TIMESTEPS = 2
SEQUENCES = ["CT1", "FLAIR", "T1", "T2"]
num_sequences = len(SEQUENCES)
sequence_keys_AB = (
    [f"{sequence}_{TIMESTEPS[0]}" for sequence in SEQUENCES] +
    [f"{sequence}_{TIMESTEPS[1]}" for sequence in SEQUENCES]
)
sequence_keys_C = [f"{sequence}_{TIMESTEPS[2]}" for sequence in SEQUENCES]
seg_keys = [f"seg_{timestep}" for timestep in TIMESTEPS]

# transforms
base_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=sequence_keys_AB + sequence_keys_C + seg_keys,
        image_only=False,
        ensure_channel_first=True
    ),
    monai.transforms.ConcatItemsd(
        keys=sequence_keys_AB,
        name="imgs_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=sequence_keys_AB),
    monai.transforms.ConcatItemsd(
        keys=sequence_keys_C,
        name="imgs_C",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=sequence_keys_C),
    monai.transforms.ConcatItemsd(
        keys=["seg_A", "seg_B"],
        name="seg_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=["seg_A", "seg_B"]),
    monai.transforms.CropForegroundd(
        keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
        source_key="imgs_AB",
    ),
    monai.transforms.Spacingd(
        keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "bilinear", "nearest", "nearest"),
    ),
    monai.transforms.ThresholdIntensityd(
        keys=["seg_AB", "seg_C"],
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="seg_C", to_onehot=NUM_CLASSES),
    # monai.transforms.Orientationd(
    #     keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
    #     axcodes="IPL"
    # )
])
train_transforms = monai.transforms.Compose([
    monai.transforms.RandAffined(
        keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
        prob=1.0,
        rotate_range=0.1,
        scale_range=0.1,
        mode=("bilinear", "bilinear", "nearest", "nearest"),
        padding_mode="zeros"
    ),
    monai.transforms.RandCropByPosNegLabeld(
        keys=["imgs_AB", "imgs_C", "seg_AB", "seg_C"],
        label_key="seg_C",
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=1,
    ),
    # images_AB and images_C have different number of channels, which leads to
    # an error when processed together by RandGaussianNoised
    monai.transforms.RandGaussianNoised(
        keys="imgs_AB",
        prob=1.0,
        mean=0,
        std=20
    ),
    monai.transforms.RandGaussianNoised(
        keys="imgs_C",
        prob=1.0,
        mean=0,
        std=20
    ),
    monai.transforms.NormalizeIntensityd(
        keys=["imgs_AB", "imgs_C"],
        channel_wise=True
    )
])
eval_transforms = monai.transforms.Compose([
    monai.transforms.NormalizeIntensityd(
        keys=["imgs_AB", "imgs_C"],
        channel_wise=True
    )
])

# model
MODEL_KWARGS_AB2C = {
    "in_channels": NUM_INPUT_TIMESTEPS * num_sequences + NUM_INPUT_TIMESTEPS,
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
checkpoint_dir = os.path.join("checkpoints", "with_seg_AB")
MODEL_NAME = "cascade_net"
logs_dir = os.path.join("logs", "with_seg_AB")
train_logs_path = os.path.join(logs_dir, "train_logs.json")
pred_loss_plot_path = os.path.join(logs_dir, "pred_loss_plot.png")
cycle_loss_plot_path = os.path.join(logs_dir, "cycle_loss_plot.png")
kl_loss_plot_path = os.path.join(logs_dir, "kl_loss_plot.png")
metric_plot_path = os.path.join(logs_dir, "metric_plot.png")
test_logs_path = os.path.join(logs_dir, "test_logs.json")