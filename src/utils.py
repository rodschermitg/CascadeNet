import os

import matplotlib
import matplotlib.pyplot as plt

try:
    import config
except ModuleNotFoundError:
    from src import config


def create_log_plots(
    train_logs,
    output_path,
    train_crit_keys=[],
    val_crit_keys=[],
    y_labels=None,
    ylim=None
):
    if isinstance(train_crit_keys, str):
        train_crit_keys = [train_crit_keys]
    if isinstance(val_crit_keys, str):
        val_crit_keys = [val_crit_keys]

    rows = len(train_crit_keys) if train_crit_keys else len(val_crit_keys)
    if train_crit_keys and val_crit_keys:
        assert len(train_crit_keys) == len(val_crit_keys)
        cols = 2
    else:
        cols = 1

    if y_labels:
        assert rows == len(y_labels)

    matplotlib.use("Agg")
    plt.style.use("ggplot")

    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)

    for row in range(rows):
        for fold in range(config.FOLDS):
            if len(train_crit_keys) == 0 and len(val_crit_keys) == 1:
                val_ax = ax
            elif len(train_crit_keys) == 1 and len(val_crit_keys) == 0:
                train_ax = ax
            elif len(train_crit_keys) == 0 and len(val_crit_keys) != 0:
                val_ax = ax[row]
            elif len(train_crit_keys) != 0 and len(val_crit_keys) == 0:
                train_ax = ax[row]
            elif len(train_crit_keys) == 1 and len(val_crit_keys) == 1:
                train_ax = ax[0]
                val_ax = ax[1]
            else:
                train_ax = ax[row, 0]
                val_ax = ax[row, 1]

            if train_crit_keys:
                train_y_values = train_logs[train_crit_keys[row]][
                    f"fold{fold}"
                ][:train_logs["best_epoch"][f"fold{fold}"]+1]
                train_x_values = list(range(len(train_y_values)))
                train_ax.plot(train_x_values, train_y_values)
            if val_crit_keys:
                val_y_values = train_logs[val_crit_keys[row]][
                    f"fold{fold}"
                ][:train_logs["best_epoch"][f"fold{fold}"]+1]
                val_x_values = list(range(
                    0,
                    config.VAL_INTERVAL*len(val_y_values),
                    config.VAL_INTERVAL)
                )
                val_ax.plot(val_x_values, val_y_values)

            if y_labels and train_crit_keys:
                train_ax.set_ylabel(y_labels[row])
            elif y_labels and val_crit_keys:
                val_ax.set_ylabel(y_labels[row])

            if row == 0 and train_crit_keys:
                train_ax.set_title("train")
            if row == 0 and val_crit_keys:
                val_ax.set_title("val")

            if row == rows - 1 and train_crit_keys:
                train_ax.set_xlabel("epoch")
            if row == rows - 1 and val_crit_keys:
                val_ax.set_xlabel("epoch")

            if ylim and train_crit_keys:
                train_ax.set_ylim(ylim)
            if ylim and val_crit_keys:
                val_ax.set_ylim(ylim)

            fig.legend(
                [f"fold{fold}" for fold in range(config.FOLDS)],
                loc="lower right"
            )

    plt.subplots_adjust(right=0.85)
    plt.savefig(output_path)
    plt.close()


def create_slice_plots(
    imgs,
    slice_dim=0,
    title=None,
    labels=None,
    num_slices=10
):
    num_imgs = len(imgs)

    fig, ax = plt.subplots(num_slices, num_imgs)
    for col, img in enumerate(imgs):
        total_slices = img.shape[slice_dim+1]
        slice_stride = total_slices // num_slices

        img = img.permute(1, 2, 3, 0)
        for row in range(num_slices):
            ax[row, col].axis("off")

            slice_idx = (row + 1) * slice_stride
            slice_idx = min(slice_idx, total_slices-1)  # avoid IndexError

            if slice_dim == 0:
                img_slice = img[slice_idx, :, :, :]
            elif slice_dim == 1:
                img_slice = img[:, slice_idx, :, :]
            elif slice_dim == 2:
                img_slice = img[:, :, slice_idx, :]

            ax[row, col].imshow(
                img_slice,
                vmin=0 if len(img_slice.unique()) == 1 else None,
                vmax=1 if len(img_slice.unique()) == 1 else None
            )

            if row == 0 and labels is not None:
                ax[row, col].set_title(labels[col])

    fig.suptitle(title)
    plt.show()


def get_patient_name(file_path):
    start_idx = file_path.find("Patient")
    end_idx = file_path.find(os.sep, start_idx)
    patient_name = file_path[start_idx:end_idx]

    return patient_name
