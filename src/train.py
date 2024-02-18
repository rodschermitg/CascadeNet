import itertools
import json
import os
import time

import monai
import sklearn.model_selection
import torch

import config
import models
import utils


monai.utils.set_determinism(config.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
# entire dataset is first stored into CacheDataset and later extracted into
# separate Subsets during cross validation
dataset = monai.data.CacheDataset(
    data=data["train"],
    transform=config.base_transforms,
    num_workers=num_workers
)

net_A2B = models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS_A2B).to(device)
net_A2B.init_weights(torch.nn.init.kaiming_uniform_, 0)
net_A2B.init_bias(torch.nn.init.constant_, 0)
net_B2A = models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS_B2A).to(device)
net_B2A.init_weights(torch.nn.init.kaiming_uniform_, 0)
net_B2A.init_bias(torch.nn.init.constant_, 0)

discretize = monai.transforms.AsDiscrete(
    argmax=True,
    to_onehot=config.NUM_CLASSES
)

optimizer = torch.optim.Adam(
    itertools.chain(net_A2B.parameters(), net_B2A.parameters()),
    lr=config.LR
)
scaler = torch.cuda.amp.GradScaler(enabled=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    min_lr=config.LR/10,
    verbose=True
)

loss_fn_pred = torch.nn.NLLLoss(reduction="mean")
loss_fn_cycle = torch.nn.L1Loss(reduction="mean")
dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    include_background=False,
    metric_name=("precision", "recall")
)

k_fold = sklearn.model_selection.KFold(
    n_splits=config.FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE
)
fold_indices = k_fold.split(dataset)

best_fold = -1
best_epoch = -1
best_dice = -1

train_logs = {
    "total_train_time": -1,
    "fold_indices": {f"fold{i}": {} for i in range(config.FOLDS)},
    "mean_train_loss_pred": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_kl_A2B": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_cycle": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_kl_B2A": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_loss_pred": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_loss_cycle": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_dice": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_precision": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_recall": {f"fold{i}": [] for i in range(config.FOLDS)},
    "best_fold": best_fold,
    "best_epoch": best_epoch,
    "best_dice": best_dice
}

print("Training network")
train_start_time = time.perf_counter()

for fold, (train_indices, val_indices) in enumerate(fold_indices):
    train_logs["fold_indices"][f"fold{fold}"] = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist()
    }

    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)

    train_data = monai.data.Dataset(train_data, config.train_transforms)
    val_data = monai.data.CacheDataset(
        val_data,
        config.eval_transforms,
        num_workers=num_workers,
        progress=False
    )

    train_dataloader = monai.data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = monai.data.DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for epoch in range(config.EPOCHS):
        net_A2B.train()
        net_B2A.train()

        epoch_train_loss_pred = 0
        epoch_train_loss_kl_A2B = 0
        epoch_train_loss_cycle = 0
        epoch_train_loss_kl_B2A = 0

        for iter, train_batch in enumerate(train_dataloader):
            train_real_A = train_batch["images_A"].to(device)
            train_real_B = train_batch["images_B"].to(device)
            train_label_B = train_batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=False):
                # net_A2B
                train_pred_B, train_loss_kl_A2B = net_A2B(
                    train_real_A,
                    train_label_B
                )
                train_loss_pred = loss_fn_pred(
                    train_pred_B,
                    torch.argmax(train_label_B, dim=1)  # decode one-hot labels
                )

                # if net_A2B.unet.save_decoder_features:
                #     decoder_features = net_A2B.get_processed_decoder_features(
                #         config.PATCH_SIZE
                #     )
                #     train_loss_ds = sum([
                #         loss_fn_pred(
                #             feat,
                #             torch.argmax(train_label_B, dim=1)
                #         ) * (1/2)**d
                #         for d, feat in enumerate(decoder_features, 1)
                #     ])

                # net_B2A
                mean = train_pred_B.mean(dim=(2, 3, 4), keepdim=True)
                std = train_pred_B.std(dim=(2, 3, 4), keepdim=True)
                train_pred_B = (train_pred_B - mean) / std

                train_rec_A, train_loss_kl_B2A = net_B2A(
                    torch.cat((train_pred_B, train_real_B), dim=1),
                    train_real_A
                )

                train_loss_cycle = loss_fn_cycle(train_real_A, train_rec_A)

            train_loss = (
                train_loss_pred +
                config.KL_WEIGHT*train_loss_kl_A2B +
                config.KL_WEIGHT*train_loss_kl_B2A +
                config.CYCLE_WEIGHT*train_loss_cycle
            )
            # if net_A2B.unet.save_decoder_features:
            #     train_loss += train_loss_ds

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss_pred += train_loss_pred.item()
            epoch_train_loss_kl_A2B += train_loss_kl_A2B.item()
            epoch_train_loss_cycle += train_loss_cycle.item()
            epoch_train_loss_kl_B2A += train_loss_kl_B2A.item()

            if iter == 0:
                print("")
            if (iter + 1) % config.DISPLAY_INTERVAL == 0:
                print(
                    f"Fold [{fold+1:1}/{config.FOLDS}], "
                    f"Epoch [{epoch+1:3}/{config.EPOCHS}], "
                    f"Iter [{iter+1:2}/{len(train_dataloader)}], "
                    f"Prediction loss: {train_loss_pred.item():.4f}, "
                    f"KL loss (net_A2B): {train_loss_kl_A2B.item():.4f}, "
                    f"Cycle loss: {train_loss_cycle.item():.4f}, "
                    f"KL loss (net_B2A): {train_loss_kl_B2A.item():.4f}"
                )

        mean_train_loss_pred = epoch_train_loss_pred / len(train_dataloader)
        mean_train_loss_kl_A2B = (
            epoch_train_loss_kl_A2B / len(train_dataloader)
        )
        mean_train_loss_cycle = epoch_train_loss_cycle / len(train_dataloader)
        mean_train_loss_kl_B2A = (
            epoch_train_loss_kl_B2A / len(train_dataloader)
        )

        train_logs["mean_train_loss_pred"][f"fold{fold}"].append(
            mean_train_loss_pred
        )
        train_logs["mean_train_loss_kl_A2B"][f"fold{fold}"].append(
            mean_train_loss_kl_A2B
        )
        train_logs["mean_train_loss_cycle"][f"fold{fold}"].append(
            mean_train_loss_cycle
        )
        train_logs["mean_train_loss_kl_B2A"][f"fold{fold}"].append(
            mean_train_loss_kl_B2A
        )

        print(f"Mean train prediction loss: {mean_train_loss_pred:.4f}")
        print(f"Mean train KL loss (net_A2B): {mean_train_loss_kl_A2B:.4f}")
        print(f"Mean train cycle loss: {mean_train_loss_cycle:.4f}")
        print(f"Mean train KL loss (net_B2A): {mean_train_loss_kl_B2A:.4f}")

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            net_A2B.eval()
            net_B2A.eval()

            epoch_val_loss_pred = 0
            epoch_val_loss_cycle = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_real_A = val_batch["images_A"].to(device)
                    val_real_B = val_batch["images_B"].to(device)
                    val_label_B = val_batch["label"].to(device)

                    with torch.cuda.amp.autocast():
                        # net_A2B
                        val_pred_B = monai.inferers.sliding_window_inference(
                            inputs=val_real_A,
                            roi_size=config.PATCH_SIZE,
                            sw_batch_size=config.BATCH_SIZE,
                            predictor=net_A2B
                        )
                        epoch_val_loss_pred += loss_fn_pred(
                            val_pred_B,
                            torch.argmax(val_label_B, dim=1)
                        ).item()

                        # net_B2A
                        mean = val_pred_B.mean(dim=(2, 3, 4), keepdim=True)
                        std = val_pred_B.std(dim=(2, 3, 4), keepdim=True)

                        val_rec_A = monai.inferers.sliding_window_inference(
                            inputs=torch.cat(
                                ((val_pred_B-mean)/std, val_real_B),
                                dim=1
                            ),
                            roi_size=config.PATCH_SIZE,
                            sw_batch_size=config.BATCH_SIZE,
                            predictor=net_B2A
                        )

                        epoch_val_loss_cycle += loss_fn_cycle(
                            val_real_A,
                            val_rec_A
                        ).item()

                    # store discretized batches in lists for metric functions
                    val_pred_B = [
                        discretize(pred)
                        for pred in monai.data.decollate_batch(val_pred_B)
                    ]
                    val_label_B = monai.data.decollate_batch(val_label_B)
                    # metric results are stored internally
                    dice_fn(val_pred_B, val_label_B)
                    confusion_matrix_fn(val_pred_B, val_label_B)

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

            if mean_val_dice > best_dice:
                print("New best dice, serializing model to disk")

                best_fold = fold
                best_epoch = epoch
                best_dice = mean_val_dice

                train_logs["best_fold"] = best_fold
                train_logs["best_epoch"] = best_epoch
                train_logs["best_dice"] = best_dice

                torch.save(
                    {
                        "net_A2B_state_dict": net_A2B.state_dict(),
                        "net_B2A_state_dict": net_B2A.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "fold": best_fold,
                        "epoch": best_epoch
                    },
                    os.path.join(config.model_dir, f"{config.MODEL_NAME}.tar")
                )
            else:
                print(
                    f"Current best dice: {best_dice:.4f} at fold "
                    f"{best_fold+1}, epoch {best_epoch+1}"
                )

            scheduler.step(mean_val_loss_pred)

            utils.create_log_plots(
                y_list=[
                    utils.concat_logs(train_logs["mean_train_loss_pred"]),
                    utils.concat_logs(train_logs["mean_val_loss_pred"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["train prediction loss", "val prediction loss"],
                output_path=config.pred_loss_plot_path,
                title="mean training/validation prediction loss per epoch",
                y_label="Loss"
            )
            utils.create_log_plots(
                y_list=[
                    utils.concat_logs(train_logs["mean_train_loss_cycle"]),
                    utils.concat_logs(train_logs["mean_val_loss_cycle"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["train cycle loss", "val cycle loss"],
                output_path=config.cycle_loss_plot_path,
                title="mean cycle loss per epoch",
                y_label="Loss"
            )
            utils.create_log_plots(
                y_list=[
                    utils.concat_logs(train_logs["mean_train_loss_kl_A2B"]),
                    utils.concat_logs(train_logs["mean_train_loss_kl_B2A"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["train KL loss (net_A2B)", "train KL loss (net_B2A)"],
                output_path=config.kl_loss_plot_path,
                title="mean training KL loss per epoch",
                y_label="Loss"
            )
            utils.create_log_plots(
                y_list=[
                    utils.concat_logs(train_logs["mean_val_dice"]),
                    utils.concat_logs(train_logs["mean_val_precision"]),
                    utils.concat_logs(train_logs["mean_val_recall"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["val dice", "val precision", "val recall"],
                output_path=config.val_metric_plot_path,
                best_fold=best_fold,
                best_epoch=best_epoch,
                best_dice=best_dice,
                title="mean validation metrics per epoch",
                y_label="Metric"
            )

            with open(config.train_logs_path, "w") as train_logs_file:
                json.dump(train_logs, train_logs_file, indent=4)

    if config.SAVE_MODEL_EACH_FOLD:
        print(f"Fold {fold+1} completed, serializing model to disk")
        torch.save(
            {
                "net_A2B_state_dict": net_A2B.state_dict(),
                "net_B2A_state_dict": net_B2A.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "fold": fold,
                "epoch": epoch
            },
            os.path.join(
                config.model_dir, f"{config.MODEL_NAME}_fold{fold}.tar"
            )
        )

train_end_time = time.perf_counter()
total_train_time = train_end_time - train_start_time
train_logs["total_train_time"] = total_train_time
print(f"\nTraining finished after {total_train_time:.0f}s")

with open(config.train_logs_path, "w") as train_logs_file:
    json.dump(train_logs, train_logs_file, indent=4)


# TODO
# - add hyperparameter optimization?
# - increase patch_size?
