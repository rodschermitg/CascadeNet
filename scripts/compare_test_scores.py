"""run from project root: python3 -m scripts.compare_test_scores"""
import importlib
import json

import matplotlib
import matplotlib.pyplot as plt

from src import config_compare


config_dict = {
    model_key: importlib.import_module(f"src.{config_name}")
    for model_key, config_name in config_compare.CONFIG_NAME_DICT.items()
}

matplotlib.use("TkAgg")
plt.style.use("ggplot")

for model_name, config in config_dict.items():
    with open(config.test_logs_path, "r") as test_logs_file:
        test_logs = json.load(test_logs_file)

    patient_names = test_logs["individual"].keys()
    scores = [
        test_logs["individual"][patient_name]["dice"]
        for patient_name in patient_names
    ]

    plt.scatter(
        patient_names,
        scores,
        marker="_",
        linewidths=3,
        label=model_name
    )

plt.ylabel("Dice")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend(reverse=True)
plt.tight_layout()
plt.show()
