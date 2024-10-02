import torch
import gc
import os

import torch.nn as nn
import torch.optim as optim

import time

# import torch_xla.core.xla_model as xm

from baselineUNet import baselineUNet
from newUNet import newUNet
from dataset import get_loaders
from utils import save_checkpoint, save_all_predictions, save_predictions_as_imgs
from train import train_one_epoch, evaluate
from optimal_workers import get_active_config
from loss import FocalLoss, DiceLoss

ActiveConfig = get_active_config()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True  # it means that the data loader copies Tensors into CUDA pinned memory before returning them, which can speed up memory transfers from CPU to GPU


TESTING = "test" if True else ""
EXPERIMENT = "/experiment-model"
TRAIN_IMG_DIR = ActiveConfig.TRAIN_IMG_DIR
TRAIN_MASK_DIR = ActiveConfig.TRAIN_MASK_DIR
VAL_IMG_DIR = ActiveConfig.VAL_IMG_DIR
VAL_MASK_DIR = ActiveConfig.VAL_MASK_DIR

SAVED_IMAGES_DIR = ActiveConfig.SAVED_IMAGES_DIR+ EXPERIMENT
CHECKPOINT_DIR = ActiveConfig.CHECKPOINT_DIR + EXPERIMENT
WRITER_DIR = ActiveConfig.WRITER_DIR + EXPERIMENT


def experiment2(experiment_name="test",lr=1e-3, model_name="newUNet", num_filters=64, num_depth=5, activation_fn="relu", loss_fn="bce", batch_size=16, num_workers=0, max_epochs=1,
         optimizer_name="adam", scaler_name="True", 
         bce_weight = None, focal_gamma=2):

    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(WRITER_DIR, exist_ok=True)

    if model_name == "newUNet":
        model = newUNet(in_channels=6, out_channels=1, depth=num_depth, num_filters=num_filters, activation=activation_fn).to(DEVICE)
    elif model_name == "baselineUNet":
        model = baselineUNet(in_channels=6, out_channels=1, activation=activation_fn).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model class: {model_name}")

    model_name = model.__class__.__name__
    current_checkpoint_dir = CHECKPOINT_DIR + "/" + model_name
    os.makedirs(current_checkpoint_dir, exist_ok=True)
    print("Model is set.")

    # reductions are 'mean' by default:
    if loss_fn == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
    elif loss_fn == "focal":
        loss_fn = FocalLoss(gamma=focal_gamma)
    elif loss_fn == "dice":
        loss_fn = DiceLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    print("Loss function is set.")

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    print("Optimizer is set.")

    if scaler_name == "True":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    print("Scaler is set.")

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        batch_size,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        num_workers,
        PIN_MEMORY,
    )
    print("Data loaders are set.")

    current_writer_dir = WRITER_DIR + "/" + model_name
    os.makedirs(WRITER_DIR + "/" + model_name, exist_ok=True)

    f1_best = 0
    validation_freq = 2

    f1_train_list = [] # (index, f1)
    f1_val_list = [] 
    iou_train_list = []
    iou_val_list = []
    precision_train_list = []
    precision_val_list = []
    recall_train_list = []
    recall_val_list = []
    acc_train_list = []
    acc_val_list = []

    start_time = time.time()
    for epoch in range(max_epochs):
        print(f"epoch {epoch}")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        # check accuracy
        if (epoch +1) % validation_freq == 0:
            f1_train, iou_train, precision_train, recall_train, acc_train = evaluate(train_loader, model, device=DEVICE)
            f1_val, iou_val, precision_val, recall_val, acc_val = evaluate(val_loader, model, device=DEVICE)

            f1_train_list.append((epoch, f1_train))
            f1_val_list.append((epoch, f1_val))
            iou_train_list.append((epoch, iou_train))
            iou_val_list.append((epoch, iou_val))
            precision_train_list.append((epoch, precision_train))
            precision_val_list.append((epoch, precision_val))
            recall_train_list.append((epoch, recall_train))
            recall_val_list.append((epoch, recall_val))
            acc_train_list.append((epoch, acc_train))
            acc_val_list.append((epoch, acc_val))

            if f1_val > f1_best:
                f1_best = f1_val
                checkpoint = {
                    "state_dict": model.state_dict(),  # model weights
                    "optimizer": optimizer.state_dict(),  # optimizer state, e.g. learning rate, momentum, etc.
                    "scaler": (
                        scaler.state_dict() if scaler is not None else None
                    ),  # scaler state, e.g. scale, offset
                }
                save_checkpoint(
                    checkpoint,
                    current_checkpoint_dir + "/" + experiment_name, experiment_name+"_best"
                )
            elif epoch == max_epochs - 1:
                checkpoint = {
                    "state_dict": model.state_dict(),  
                    "optimizer": optimizer.state_dict(),  
                    "scaler": (
                        scaler.state_dict() if scaler is not None else None
                    )
                }
                save_checkpoint(
                    checkpoint,
                    current_checkpoint_dir
                    + "/"
                    + experiment_name,
                    experiment_name + "_last"
                )

    end_time = time.time()

    # save_predictions_as_imgs(val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE)
    # save score lists to a file
    with open(current_writer_dir + "/" + experiment_name + "_f1_train_list.csv", "w") as f:
        for item in f1_train_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_f1_val_list.csv", "w") as f:
        for item in f1_val_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_iou_train_list.csv", "w") as f:
        for item in iou_train_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_iou_val_list.csv", "w") as f:
        for item in iou_val_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_precision_train_list.csv", "w") as f:
        for item in precision_train_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_precision_val_list.csv", "w") as f:
        for item in precision_val_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_recall_train_list.csv", "w") as f:
        for item in recall_train_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_recall_val_list.csv", "w") as f:
        for item in recall_val_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_acc_train_list.csv", "w") as f:
        for item in acc_train_list:
            f.write(f"{item[0]},{item[1]}\n")
    with open(current_writer_dir + "/" + experiment_name + "_acc_val_list.csv", "w") as f:
        for item in acc_val_list:
            f.write(f"{item[0]},{item[1]}\n")

    save_predictions_as_imgs(val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE)
    save_all_predictions(val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE)

    return f1_best, end_time - start_time


if __name__ == "__main__":  # to not have issues when running NUM_WORKERS on Windows
    gc.collect()
    experiment2()
    gc.collect()
