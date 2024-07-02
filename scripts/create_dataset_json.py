"""run from project root: python3 -m scripts.create_dataset_json"""
import os
import itertools
import json

import monai
import sklearn.model_selection

from src import config


def get_week_num(week_path):
    start_idx = week_path.find("week-") + len("week-")
    end_idx = start_idx + 3
    week_num = int(week_path[start_idx:end_idx])

    return week_num


def get_tumor_volume(week_path):
    transform = monai.transforms.Compose([
        monai.transforms.LoadImage(
            image_only=True,
            ensure_channel_first=True
        ),
        monai.transforms.CropForeground(),
        monai.transforms.ThresholdIntensity(
            threshold=1,
            above=False,
            cval=1
        )
    ])

    seg_path = os.path.join(week_path, "seg_mask.nii.gz")
    tumor_volume = transform(seg_path).sum()

    return tumor_volume


def create_patient_dict(week_paths, max_week_num):
    patient_dict = {}
    for timestep_idx, timestep in enumerate(config.TIMESTEPS):
        for image_key in config.SEQUENCES + ["tissue_seg", "seg"]:
            if image_key == "tissue_seg":
                img_filename = "tissue_seg.nii.gz"
            elif image_key == "seg":
                img_filename = "seg_mask.nii.gz"
            else:
                img_filename = f"{image_key.lower()}_skull_strip.nii.gz"
            img_path = os.path.join(week_paths[timestep_idx], img_filename)
            patient_dict[f"{image_key}_{timestep}"] = img_path

    week_num_A = get_week_num(week_paths[0])
    week_num_B = get_week_num(week_paths[1])
    week_num_C = get_week_num(week_paths[2])

    patient_dict["time_diff_AC"] = (week_num_C - week_num_A) / max_week_num
    patient_dict["time_diff_BC"] = (week_num_C - week_num_B) / max_week_num

    return patient_dict


MIN_TIME_DIFF = 2
MAX_WEEK_NUM = 255

dataset_dict = {"train": {}, "val": {}, "test": []}

# test dataset
for patient_dir in sorted(os.listdir(config.test_data_dir)):
    patient_path = os.path.join(config.test_data_dir, patient_dir)
    week_dirs = sorted(os.listdir(patient_path))
    week_combs = list(itertools.combinations(week_dirs, 3))

    for week_comb in week_combs:
        week_paths = [
            os.path.join(patient_path, week_dir) for week_dir in week_comb
        ]

        week_num_A = get_week_num(week_paths[0])
        week_num_B = get_week_num(week_paths[1])
        week_num_C = get_week_num(week_paths[2])

        tumor_volume_A = get_tumor_volume(week_paths[0])
        tumor_volume_B = get_tumor_volume(week_paths[1])
        tumor_volume_C = get_tumor_volume(week_paths[2])

        if (
            week_num_B - week_num_A >= MIN_TIME_DIFF and
            week_num_C - week_num_B >= MIN_TIME_DIFF and
            tumor_volume_B > tumor_volume_A and
            tumor_volume_C > tumor_volume_B
        ):
            patient_dict = create_patient_dict(week_paths, MAX_WEEK_NUM)
            dataset_dict["test"].append(patient_dict)

    if patient_dir not in list(dataset_dict["test"][-1].values())[0]:
        week_dirs = [week_dirs[0], week_dirs[-2], week_dirs[-1]]
        week_paths = [
            os.path.join(patient_path, week_dir) for week_dir in week_dirs
        ]
        patient_dict = create_patient_dict(week_paths, MAX_WEEK_NUM)
        dataset_dict["test"].append(patient_dict)

# train/val dataset
idx_dict = {}
for patient_idx, patient_dir in enumerate(
    sorted(os.listdir(config.train_data_dir))
):
    patient_path = os.path.join(config.train_data_dir, patient_dir)
    week_dirs = sorted(os.listdir(patient_path))
    week_combs = list(itertools.combinations(week_dirs, 3))
    idx_dict[patient_idx] = []

    for week_comb in week_combs:
        week_paths = [
            os.path.join(patient_path, week_dir) for week_dir in week_comb
        ]

        week_num_A = get_week_num(week_paths[0])
        week_num_B = get_week_num(week_paths[1])
        week_num_C = get_week_num(week_paths[2])

        tumor_volume_A = get_tumor_volume(week_paths[0])
        tumor_volume_B = get_tumor_volume(week_paths[1])
        tumor_volume_C = get_tumor_volume(week_paths[2])

        if (
            week_num_B - week_num_A >= MIN_TIME_DIFF and
            week_num_C - week_num_B >= MIN_TIME_DIFF and
            tumor_volume_B > tumor_volume_A and
            tumor_volume_C > tumor_volume_B
        ):
            patient_dict = create_patient_dict(week_paths, MAX_WEEK_NUM)
            idx_dict[patient_idx].append(patient_dict)

    if not idx_dict[patient_idx]:
        week_dirs = [week_dirs[0], week_dirs[-2], week_dirs[-1]]
        week_paths = [
            os.path.join(patient_path, week_dir) for week_dir in week_dirs
        ]
        patient_dict = create_patient_dict(week_paths, MAX_WEEK_NUM)
        idx_dict[patient_idx] = [patient_dict]

k_fold = sklearn.model_selection.KFold(
    n_splits=config.FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE
)
fold_idxs = k_fold.split(idx_dict)

for fold, (train_idxs, val_idxs) in enumerate(fold_idxs):
    dataset_dict["train"][f"fold{fold}"] = [
        sample for idx in train_idxs for sample in idx_dict[idx]
    ]
    dataset_dict["val"][f"fold{fold}"] = [
        sample for idx in val_idxs for sample in idx_dict[idx]
    ]

with open(os.path.join(config.data_dir, "idx_dict.json"), "w") as file:
    json.dump(idx_dict, file, indent=4)
with open(os.path.join(config.data_dir, config.DATA_FILENAME), "w") as file:
    json.dump(dataset_dict, file, indent=4)
