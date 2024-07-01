import itertools
import json
import os
import time

import monai
import torch

import config
import models
import transforms
import utils


monai.utils.set_determinism(config.RANDOM_STATE)
print(f"Current task: {config.TASK}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)

train_transforms = monai.transforms.Compose([
    *transforms.transforms_dict[config.TASK]["base_transforms"].transforms,
    *transforms.transforms_dict[config.TASK]["train_transforms"].transforms
])
val_transforms = monai.transforms.Compose([
    *transforms.transforms_dict[config.TASK]["base_transforms"].transforms,
    *transforms.transforms_dict[config.TASK]["eval_transforms"].transforms
])

discretize = monai.transforms.AsDiscrete(
    argmax=True,
    to_onehot=config.NUM_CLASSES
)

loss_fn_pred = monai.losses.DiceCELoss(
    include_background=False,
    softmax=True,
    reduction="mean"
)
loss_fn_cycle = torch.nn.L1Loss(reduction="mean")
dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    include_background=False,
    metric_name=("precision", "recall")
)

train_logs = {
    "total_train_time": None,
    "mean_train_loss_pred": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_kl_AB2C": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_cycle": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_kl_C2AB": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_dice": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_precision": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_recall": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_loss_pred": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_loss_cycle": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_dice": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_precision": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_recall": {f"fold{i}": [] for i in range(config.FOLDS)},
    "best_fold": None,
    "best_epoch": None,
    "best_dice": -1
}

print("Training network")
train_start_time = time.perf_counter()

for fold in range(config.FOLDS):
    net_AB2C = models.ProbabilisticSegmentationNet(
        **config.NET_AB2C_KWARGS_DICT[config.TASK]
    ).to(device)
    net_AB2C.init_weights(torch.nn.init.kaiming_uniform_, 0)
    net_AB2C.init_bias(torch.nn.init.constant_, 0)
    net_C2AB = models.ProbabilisticSegmentationNet(
        **config.NET_C2AB_KWARGS
    ).to(device)
    net_C2AB.init_weights(torch.nn.init.kaiming_uniform_, 0)
    net_C2AB.init_bias(torch.nn.init.constant_, 0)

    optimizer = torch.optim.Adam(
        itertools.chain(net_AB2C.parameters(), net_C2AB.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True
    )

    train_data = monai.data.Dataset(
        data["train"][f"fold{fold}"],
        train_transforms
    )
    val_data = monai.data.Dataset(
        data["val"][f"fold{fold}"],
        val_transforms
    )

    train_dataloader = monai.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = monai.data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for epoch in range(config.EPOCHS):
        net_AB2C.train()
        net_C2AB.train()

        epoch_train_loss_pred = 0
        epoch_train_loss_kl_AB2C = 0
        epoch_train_loss_cycle = 0
        epoch_train_loss_kl_C2AB = 0

        for iter, train_batch in enumerate(train_dataloader):
            train_input_AB = train_batch[
                config.INPUT_DICT[config.TASK]
            ].to(device)
            train_real_AB = train_batch["img_AB"].to(device)
            train_real_C = train_batch["img_C"].to(device)
            train_seg_C = train_batch["seg_C"].to(device)

            with torch.cuda.amp.autocast(enabled=False):
                # net_AB2C
                train_pred_C, train_loss_kl_AB2C = net_AB2C(
                    train_input_AB,
                    train_seg_C
                )
                train_loss_pred = loss_fn_pred(train_pred_C, train_seg_C)

                # net_C2AB
                train_pred_C = torch.nn.functional.softmax(train_pred_C, dim=1)
                mean = train_pred_C.mean(dim=(2, 3, 4), keepdim=True)
                std = train_pred_C.std(dim=(2, 3, 4), keepdim=True)
                train_pred_C = (train_pred_C - mean) / std

                train_rec_AB, train_loss_kl_C2AB = net_C2AB(
                    torch.cat((train_pred_C, train_real_C), dim=1),
                    train_real_AB
                )

                train_loss_cycle = loss_fn_cycle(train_real_AB, train_rec_AB)

            train_loss = (
                train_loss_pred +
                config.KL_WEIGHT*train_loss_kl_AB2C +
                config.KL_WEIGHT*train_loss_kl_C2AB +
                config.CYCLE_WEIGHT*train_loss_cycle
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss_pred += train_loss_pred.item()
            epoch_train_loss_kl_AB2C += train_loss_kl_AB2C.item()
            epoch_train_loss_cycle += train_loss_cycle.item()
            epoch_train_loss_kl_C2AB += train_loss_kl_C2AB.item()

            # store discretized batches in lists for metric functions
            train_pred_C = [
                discretize(pred)
                for pred in monai.data.decollate_batch(train_pred_C)
            ]
            train_seg_C = monai.data.decollate_batch(train_seg_C)

            # metric results are stored internally
            dice_fn(train_pred_C, train_seg_C)
            confusion_matrix_fn(train_pred_C, train_seg_C)

            if iter == 0:
                print("")
            if (iter + 1) % config.DISPLAY_INTERVAL == 0:
                print(
                    f"Fold [{fold+1:1}/{config.FOLDS}], "
                    f"Epoch [{epoch+1:2}/{config.EPOCHS}], "
                    f"Iter [{iter+1:3}/{len(train_dataloader)}], "
                    f"Prediction loss: {train_loss_pred.item():.4f}, "
                    f"KL loss (net_AB2C): {train_loss_kl_AB2C.item():.4f}, "
                    f"Cycle loss: {train_loss_cycle.item():.4f}, "
                    f"KL loss (net_C2AB): {train_loss_kl_C2AB.item():.4f}"
                )

        mean_train_loss_pred = epoch_train_loss_pred / len(train_dataloader)
        mean_train_loss_kl_AB2C = (
            epoch_train_loss_kl_AB2C / len(train_dataloader)
        )
        mean_train_loss_cycle = epoch_train_loss_cycle / len(train_dataloader)
        mean_train_loss_kl_C2AB = (
            epoch_train_loss_kl_C2AB / len(train_dataloader)
        )
        mean_train_dice = dice_fn.aggregate().item()
        mean_train_precision = confusion_matrix_fn.aggregate()[0].item()
        mean_train_recall = confusion_matrix_fn.aggregate()[1].item()

        dice_fn.reset()
        confusion_matrix_fn.reset()

        train_logs["mean_train_loss_pred"][f"fold{fold}"].append(
            mean_train_loss_pred
        )
        train_logs["mean_train_loss_kl_AB2C"][f"fold{fold}"].append(
            mean_train_loss_kl_AB2C
        )
        train_logs["mean_train_loss_cycle"][f"fold{fold}"].append(
            mean_train_loss_cycle
        )
        train_logs["mean_train_loss_kl_C2AB"][f"fold{fold}"].append(
            mean_train_loss_kl_C2AB
        )
        train_logs["mean_train_dice"][f"fold{fold}"].append(mean_train_dice)
        train_logs["mean_train_precision"][f"fold{fold}"].append(
            mean_train_precision
        )
        train_logs["mean_train_recall"][f"fold{fold}"].append(
            mean_train_recall
        )

        print(f"Mean train prediction loss: {mean_train_loss_pred:.4f}")
        print(f"Mean train KL loss (net_AB2C): {mean_train_loss_kl_AB2C:.4f}")
        print(f"Mean train cycle loss: {mean_train_loss_cycle:.4f}")
        print(f"Mean train KL loss (net_C2AB): {mean_train_loss_kl_C2AB:.4f}")
        print(f"Mean train dice: {mean_train_dice:.4f}")
        print(f"Mean train precision: {mean_train_precision:.4f}")
        print(f"Mean train recall: {mean_train_recall:.4f}")

        if epoch == 0 or (epoch + 1) % config.VAL_INTERVAL == 0:
            net_AB2C.eval()
            net_C2AB.eval()

            epoch_val_loss_pred = 0
            epoch_val_loss_cycle = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_input_AB = val_batch[
                        config.INPUT_DICT[config.TASK]
                    ].to(device)
                    val_real_AB = val_batch["img_AB"].to(device)
                    val_real_C = val_batch["img_C"].to(device)
                    val_seg_C = val_batch["seg_C"].to(device)

                    with torch.cuda.amp.autocast():
                        # net_AB2C
                        val_pred_C = monai.inferers.sliding_window_inference(
                            val_input_AB,
                            roi_size=config.PATCH_SIZE,
                            sw_batch_size=config.BATCH_SIZE,
                            predictor=net_AB2C
                        )
                        epoch_val_loss_pred += loss_fn_pred(
                            val_pred_C,
                            val_seg_C
                        ).item()

                        # net_C2AB
                        val_pred_C = torch.nn.functional.softmax(
                            val_pred_C,
                            dim=1
                        )
                        mean = val_pred_C.mean(dim=(2, 3, 4), keepdim=True)
                        std = val_pred_C.std(dim=(2, 3, 4), keepdim=True)

                        val_rec_AB = monai.inferers.sliding_window_inference(
                            torch.cat(
                                ((val_pred_C-mean)/std, val_real_C),
                                dim=1
                            ),
                            roi_size=config.PATCH_SIZE,
                            sw_batch_size=config.BATCH_SIZE,
                            predictor=net_C2AB
                        )

                        epoch_val_loss_cycle += loss_fn_cycle(
                            val_real_AB,
                            val_rec_AB
                        ).item()

                    val_pred_C = [
                        discretize(pred)
                        for pred in monai.data.decollate_batch(val_pred_C)
                    ]
                    val_seg_C = monai.data.decollate_batch(val_seg_C)

                    dice_fn(val_pred_C, val_seg_C)
                    confusion_matrix_fn(val_pred_C, val_seg_C)

            mean_val_loss_pred = epoch_val_loss_pred / len(val_dataloader)
            mean_val_loss_cycle = epoch_val_loss_cycle / len(val_dataloader)
            mean_val_dice = dice_fn.aggregate().item()
            mean_val_precision = confusion_matrix_fn.aggregate()[0].item()
            mean_val_recall = confusion_matrix_fn.aggregate()[1].item()

            dice_fn.reset()
            confusion_matrix_fn.reset()

            train_logs["mean_val_loss_pred"][f"fold{fold}"].append(
                mean_val_loss_pred
            )
            train_logs["mean_val_loss_cycle"][f"fold{fold}"].append(
                mean_val_loss_cycle
            )
            train_logs["mean_val_dice"][f"fold{fold}"].append(mean_val_dice)
            train_logs["mean_val_precision"][f"fold{fold}"].append(
                mean_val_precision
            )
            train_logs["mean_val_recall"][f"fold{fold}"].append(
                mean_val_recall
            )

            print(f"Mean val prediction loss: {mean_val_loss_pred:.4f}")
            print(f"Mean val cycle loss: {mean_val_loss_cycle:.4f}")
            print(f"Mean val dice: {mean_val_dice:.4f}")
            print(f"Mean val precision: {mean_val_precision:.4f}")
            print(f"Mean val recall: {mean_val_recall:.4f}")

            if mean_val_dice > train_logs["best_dice"]:
                print("New best dice, serializing model to disk")

                train_logs["best_fold"] = fold
                train_logs["best_epoch"] = epoch
                train_logs["best_dice"] = mean_val_dice

                torch.save(
                    {
                        "net_AB2C_state_dict": net_AB2C.state_dict(),
                        "net_C2AB_state_dict": net_C2AB.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "fold": fold,
                        "epoch": epoch
                    },
                    os.path.join(
                        config.checkpoint_dir,
                        f"{config.MODEL_NAME}.tar"
                    )
                )
            else:
                print(
                    f"Current best dice: {train_logs['best_dice']:.4f} at fold"
                    f" {train_logs['best_fold']+1}, epoch "
                    f"{train_logs['best_epoch']+1}"
                )

            scheduler.step(mean_val_loss_pred)

            utils.create_log_plots(
                train_logs,
                output_path=config.pred_loss_plot_path,
                train_crit_keys=["mean_train_loss_pred"],
                val_crit_keys=["mean_val_loss_pred"],
                y_labels=["prediction loss"],
            )
            utils.create_log_plots(
                train_logs,
                output_path=config.cycle_loss_plot_path,
                train_crit_keys=["mean_train_loss_cycle"],
                val_crit_keys=["mean_val_loss_cycle"],
                y_labels=["cycle loss"],
            )
            utils.create_log_plots(
                train_logs,
                output_path=config.kl_loss_plot_path,
                train_crit_keys=[
                    "mean_train_loss_kl_AB2C",
                    "mean_train_loss_kl_C2AB"
                ],
                y_labels=["KL loss (net_AB2C)", "KL loss (net_C2AB)"],
            )
            utils.create_log_plots(
                train_logs,
                output_path=config.metric_plot_path,
                train_crit_keys=[
                    "mean_train_dice",
                    "mean_train_precision",
                    "mean_train_recall"
                ],
                val_crit_keys=[
                    "mean_val_dice",
                    "mean_val_precision",
                    "mean_val_recall"
                ],
                y_labels=["dice", "precision", "recall"],
            )

            with open(config.train_logs_path, "w") as train_logs_file:
                json.dump(train_logs, train_logs_file, indent=4)

    if config.SAVE_MODEL_EACH_FOLD:
        print(f"Fold {fold+1} completed, serializing model to disk")
        torch.save(
            {
                "net_AB2C_state_dict": net_AB2C.state_dict(),
                "net_C2AB_state_dict": net_C2AB.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "fold": fold,
                "epoch": epoch
            },
            os.path.join(
                config.checkpoint_dir,
                f"{config.MODEL_NAME}_fold{fold}.tar"
            )
        )

train_end_time = time.perf_counter()
total_train_time = train_end_time - train_start_time
train_logs["total_train_time"] = total_train_time
print(f"\nTraining finished after {total_train_time:.0f}s")

with open(config.train_logs_path, "w") as train_logs_file:
    json.dump(train_logs, train_logs_file, indent=4)


# TODO
# - when passing pred from net_AB2C to net_C2AB, should net_AB2C have LogSoftmax or Softmax as head?
# - compare rec range with image range -> substitute Tanh head with someting else?
