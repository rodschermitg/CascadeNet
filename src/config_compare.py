import os

import monai


# configs
CONFIG_NAME_DICT = {
    "base model": "config_base_model",
    "base model with seg_AB": "config_with_seg_AB"
}
IMGS_KEY_DICT = {
    "base model": "imgs_AB",
    "base model with seg_AB": "input_AB"
}

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
SEQUENCES = ["CT1", "FLAIR", "T1", "T2"]
num_sequences = len(SEQUENCES)
sequence_keys_AB = (
    [f"{sequence}_{TIMESTEPS[0]}" for sequence in SEQUENCES] +
    [f"{sequence}_{TIMESTEPS[1]}" for sequence in SEQUENCES]
)
seg_keys = [f"seg_{timestep}" for timestep in TIMESTEPS]

# transforms
transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=sequence_keys_AB + seg_keys,
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
        keys=["seg_A", "seg_B"],
        name="seg_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=["seg_A", "seg_B"]),
    monai.transforms.CropForegroundd(
        keys=["imgs_AB", "seg_AB", "seg_C"],
        source_key="imgs_AB",
    ),
    monai.transforms.Spacingd(
        keys=["imgs_AB", "seg_AB", "seg_C"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest", "nearest"),
    ),
    monai.transforms.ThresholdIntensityd(
        keys=["seg_AB", "seg_C"],
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="seg_C", to_onehot=NUM_CLASSES),
    # monai.transforms.Orientationd(
    #     keys=["imgs_AB", "seg_AB", "seg_C"],
    #     axcodes="IPL",
    # )
    monai.transforms.NormalizeIntensityd(keys="imgs_AB", channel_wise=True),
    monai.transforms.ConcatItemsd(
        keys=["imgs_AB", "seg_AB"],
        name="input_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys="seg_AB")
])

# output path
LOGS_DIR = "logs"
ROC_curves_plot_path = os.path.join(LOGS_DIR, "ROC_curves_plot.png")
