"""run from project root: python3 -m scripts.compare_test_scores"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt

from src import config


matplotlib.use("TkAgg")
plt.style.use("ggplot")

for task_key in config.INPUT_DICT.keys():
    test_logs_path = os.path.join("logs", task_key, "test_logs.json")
    with open(test_logs_path, "r") as test_logs_file:
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
        label=config.LABEL_DICT[task_key]
    )

plt.ylabel("Dice")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend(reverse=True)
plt.tight_layout()
plt.show()
