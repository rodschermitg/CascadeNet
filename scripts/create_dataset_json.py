"""run from project root: python3 -m scripts.create_dataset_json"""
import json
import os

from src import config


def fill_data_dict(data_dict, train=True, num_images=3):
    data_dir = config.train_data_dir if train else config.test_data_dir
    MODALITIES = ["ct1", "flair", "t1", "t2"]

    for patient_dir in sorted(os.listdir(data_dir)):
        patient_dict = {}
        week_dirs = sorted(os.listdir(os.path.join(data_dir, patient_dir)))

        # fill image paths
        for image_idx in range(num_images):
            image_dir = os.path.join(
                data_dir,
                patient_dir,
                week_dirs[image_idx],
                "DeepBraTumIA-segmentation",
                "atlas",
                "skull_strip"
            )
            for modality in MODALITIES:
                patient_dict[f"{modality}_{image_idx+1}"] = os.path.join(
                    image_dir,
                    f"{modality}_skull_strip.nii.gz"
                )

        # fill label path
        label_path = os.path.join(
            data_dir,
            patient_dir,
            week_dirs[-1],
            "DeepBraTumIA-segmentation",
            "atlas",
            "segmentation",
            "seg_mask.nii.gz"
        )
        patient_dict["label"] = label_path

        data_dict["train" if train else "test"].append(patient_dict)

    return data_dict


data_dict = {"train": [], "test": []}

data_dict = fill_data_dict(data_dict, train=True)
data_dict = fill_data_dict(data_dict, train=False)

with open(os.path.join(config.data_dir, config.DATA_FILENAME), "w") as file:
    json.dump(data_dict, file, indent=4)
