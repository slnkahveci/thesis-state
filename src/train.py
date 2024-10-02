import gc
import torch

from tqdm import tqdm
from eval import f1_and_iou



# import loss_fn from src/losses.py
# rename loss_fn to criterion


def train_one_epoch(loader, model, optimizer, criterion, scaler, DEVICE):  # one epoch

    model_name = model.__class__.__name__

    model.train()
    loop = tqdm(loader)  # progress bar
    total_loss = 0.0
    batch_count = len(loader)

    for batch_idx, (input, targets) in enumerate(loop):
        optimizer.zero_grad()
        input = input.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

 
        with torch.cuda.amp.autocast():
            predictions = model(input)   # NO SIGMOID AT THE END, LOSS FUNC SHOULD APPLY IT
            loss = criterion(predictions, targets)
            total_loss += loss.item()

        # backward

        if scaler is not None:
            scaler.scale(loss).backward() 
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # Free up memory after each batch
        #del input, targets, predictions, loss
        #gc.collect()  # is it needed here?
        #torch.cuda.empty_cache()
        # exit loop after one batch

    return total_loss/batch_count


def evaluate(loader, model, device): #check_accuracy

    f1_vals = 0.0
    iou_vals = 0.0
    precision_vals = 0.0
    recall_vals = 0.0
    acc_vals = 0.0

    threshold = 0.5

    batch_count = len(loader)
    model.eval()
    print("Getting metrics...")
    with torch.no_grad():  # no need to calculate gradients, it is not training. Saves memory
        for x, y in loader:

            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()

            f1, iou, precision, recall, acc = f1_and_iou(preds, y)
            f1_vals += f1
            iou_vals += iou
            precision_vals += precision
            recall_vals += recall
            acc_vals += acc

    f1_score = f1_vals / batch_count
    iou_score = iou_vals / batch_count
    precision_score = precision_vals / batch_count
    recall_score = recall_vals / batch_count
    acc_score = acc_vals / batch_count

    return f1_score, iou_score, precision_score, recall_score, acc_score
