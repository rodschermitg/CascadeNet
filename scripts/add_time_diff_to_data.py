"""run from project root: python3 -m scripts.add_time_diff_to_data"""
import json
import os

from src import config


def get_week_num(file_path):
    start_idx = file_path.find("week")
    end_idx = file_path.find(os.sep, start_idx)
    week_name = file_path[start_idx:end_idx]
    week_num = int(week_name[5:8])
    return week_num


with open(os.path.join(config.data_dir, config.DATA_FILENAME), "r") as file:
    data_dict = json.load(file)

max_week_num = 0
for dataset_key in data_dict.keys():
    for patient in data_dict[dataset_key]:
        week_num = get_week_num(patient["seg_C"])
        if max_week_num < week_num:
            max_week_num = week_num

for dataset_key in data_dict.keys():
    for patient in data_dict[dataset_key]:
        timestep_dict = {}
        for timestep in config.TIMESTEPS:
            timestep_dict[timestep] = (
                get_week_num(patient[f"seg_{timestep}"]) / max_week_num
            )
        patient["time_diff_AC"] = timestep_dict["C"] - timestep_dict["A"]
        patient["time_diff_BC"] = timestep_dict["C"] - timestep_dict["B"]

with open(os.path.join(config.data_dir, config.DATA_FILENAME), "w") as file:
    json.dump(data_dict, file, indent=4)
