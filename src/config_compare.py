import os

import monai


# configs
CONFIG_NAME_DICT = {
    "base model": "config_base_model",
    "base model with label_AB": "config_with_label_AB"
}
IMAGES_KEY_DICT = {
    "base model": "images_AB",
    "base model with label_AB": "input_AB"
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
label_keys = [f"label_{timestep}" for timestep in TIMESTEPS]

# transforms
transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=sequence_keys_AB + label_keys,
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
        keys=["label_A", "label_B"],
        name="label_AB",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=["label_A", "label_B"]),
    monai.transforms.CropForegroundd(
        keys=["images_AB", "label_AB", "label_C"],
        source_key="images_AB",
    ),
    monai.transforms.Spacingd(
        keys=["images_AB", "label_AB", "label_C"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest", "nearest"),
    ),
    monai.transforms.ThresholdIntensityd(
        keys=["label_AB", "label_C"],
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="label_C", to_onehot=NUM_CLASSES),
    # monai.transforms.Orientationd(
    #     keys=["images_AB", "label_AB", "label_C"],
    #     axcodes="IPL",
    # ),
    monai.transforms.NormalizeIntensityd(keys="images_AB", channel_wise=True),
    monai.transforms.ConcatItemsd(
        keys=["images_AB", "label_AB"],
        name="input_AB",
        dim=0
    )
])

# output path
LOGS_DIR = "logs"
ROC_curves_plot_path = os.path.join(LOGS_DIR, "ROC_curves_plot.png")
