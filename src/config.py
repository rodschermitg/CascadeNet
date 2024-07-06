import os


# task
TASK = "base_model"
# TASK = "with_tissue_seg_AB"
# TASK = "with_seg_AB"
# TASK = "with_time_diff"
# TASK = "with_tissue_seg_seg_time_diff_AB"
INPUT_DICT = {
    "base_model": "img_AB",
    "with_tissue_seg_AB": "img_tissue_seg_AB",
    "with_seg_AB": "img_seg_AB",
    "with_time_diff": "img_time_diff_AB",
    "with_tissue_seg_seg_time_diff_AB": "img_tissue_seg_seg_time_diff_AB"
}
LABEL_DICT = {
    "base_model": "base model",
    "with_tissue_seg_AB": "base model with tissue_seg_AB",
    "with_seg_AB": "base model with seg_AB",
    "with_time_diff": "base model with time information",
    "with_tissue_seg_seg_time_diff_AB": "with tissue seg, seg_AB and time diff"
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
TEST_SIZE = 0.1
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 2
TIMESTEPS = ["A", "B", "C"]
NUM_INPUTS = 2
SEQUENCES = ["CT1", "FLAIR", "T1", "T2"]
num_sequences = len(SEQUENCES)
sequence_keys = [
    [f"{sequence}_{timestep}" for sequence in SEQUENCES]
    for timestep in TIMESTEPS
]
tissue_seg_keys = [f"tissue_seg_{timestep}" for timestep in TIMESTEPS]
seg_keys = [f"seg_{timestep}" for timestep in TIMESTEPS]

# model
NET_AB2C_KWARGS_DICT = {
    "base_model": {
        "in_channels": NUM_INPUTS * num_sequences,
        "out_channels": NUM_CLASSES,
        "latent_size": 3,
        "temperature": 0.28,
        "prior_kwargs": {
            "n_components": 9
        },
        "posterior_kwargs": {
            "n_components": 9
        }
    },
    "with_tissue_seg_AB": {
        "in_channels": (
            NUM_INPUTS * num_sequences + NUM_INPUTS
        ),
        "out_channels": NUM_CLASSES,
        "latent_size": 3,
        "temperature": 0.28,
        "prior_kwargs": {
            "n_components": 9
        },
        "posterior_kwargs": {
            "n_components": 9
        }
    },
    "with_seg_AB": {
        "in_channels": (
            NUM_INPUTS * num_sequences + NUM_INPUTS
        ),
        "out_channels": NUM_CLASSES,
        "latent_size": 3,
        "temperature": 0.28,
        "prior_kwargs": {
            "n_components": 9
        },
        "posterior_kwargs": {
            "n_components": 9
        }
    },
    "with_time_diff": {
        "in_channels": (
            NUM_INPUTS * num_sequences + NUM_INPUTS
        ),
        "out_channels": NUM_CLASSES,
        "latent_size": 3,
        "temperature": 0.28,
        "prior_kwargs": {
            "n_components": 9
        },
        "posterior_kwargs": {
            "n_components": 9
        }
    },
    "with_tissue_seg_seg_time_diff_AB": {
        "in_channels": (
            NUM_INPUTS * num_sequences + 3 * NUM_INPUTS
        ),
        "out_channels": NUM_CLASSES,
        "latent_size": 3,
        "temperature": 0.28,
        "prior_kwargs": {
            "n_components": 9
        },
        "posterior_kwargs": {
            "n_components": 9
        }
    }
}
NET_C2AB_KWARGS = {
    "in_channels": num_sequences + NUM_CLASSES,
    "out_channels": 2 * num_sequences,
    "latent_size": 3,
    "temperature": 0.28,
    "prior_kwargs": {
        "n_components": 9
    },
    "posterior_kwargs": {
        "n_components": 9
    }
}

# train
LR = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 1
STOPPING_PATIENCE = 2
FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 2
VAL_INTERVAL = 1
DISPLAY_INTERVAL = 10
CYCLE_WEIGHT = 1
KL_WEIGHT = 1

# output path
checkpoint_dir = os.path.join("checkpoints", TASK)
MODEL_NAME = "cascade_net"
logs_dir = os.path.join("logs", TASK)
train_logs_path = os.path.join(logs_dir, "train_logs.json")
pred_loss_plot_path = os.path.join(logs_dir, "pred_loss_plot.png")
cycle_loss_plot_path = os.path.join(logs_dir, "cycle_loss_plot.png")
kl_loss_plot_path = os.path.join(logs_dir, "kl_loss_plot.png")
metric_plot_path = os.path.join(logs_dir, "metric_plot.png")
cv_fold_logs_path = os.path.join(logs_dir, "cv_fold_logs.json")
test_logs_path = os.path.join(logs_dir, "test_logs.json")
ROC_curves_plot_path = os.path.join("logs", "ROC_curves_plot.png")
