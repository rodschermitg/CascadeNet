"""run from project root: python3 -m scripts.plot_week_count"""
import os

import matplotlib.pyplot as plt

from src import config


PLOT_RAW_DATA = False
XTICK_IDX = [0, 15, 46, 84, 255]

data_dir = config.raw_data_dir if PLOT_RAW_DATA else config.train_data_dir
week_count_dict = {}
max_week_num = -1

for patient_dir in os.listdir(data_dir):
    for week_dir in os.listdir(os.path.join(data_dir, patient_dir)):
        week_path = os.path.join(data_dir, patient_dir, week_dir)
        if os.path.isdir(week_path):
            # get rid of duplicate suffixes
            week_dir = week_dir[:8]

            # count weeks
            week_count_dict[week_dir] = week_count_dict.get(week_dir, 0) + 1

            # get max week num
            week_num = int(week_dir[5:8])
            max_week_num = max(max_week_num, week_num)

# fill unencountered weeks with 0
for week_num in range(max_week_num):
    week = f"week-{week_num:03}"
    if week not in week_count_dict.keys():
        week_count_dict[week] = 0

# extract dict keys and vals into corresponding sorted lists
week_keys = sorted(week_count_dict.keys())
counts = [week_count_dict[week_key] for week_key in week_keys]

# make bar plot
plt.bar(week_keys, counts)
xtick_idx = XTICK_IDX if PLOT_RAW_DATA else XTICK_IDX[:-2]
xtick_labels = [f"week-{week_num:03}" for week_num in xtick_idx]
plt.xticks(xtick_idx, xtick_labels, rotation=45)
plt.tight_layout()
plt.show()

# # print list contents for sanity check
# for week, count in zip(weeks, counts):
#     if week in xtick_labels:
#         print(f"count for {week}: {count} (label displayed)")
#     else:
#         print(f"count for {week}: {count}")
