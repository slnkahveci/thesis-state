import torch
import torchvision
import matplotlib.pyplot as plt
from IPython.display import display


def f1_and_iou(preds, labels):

    tn = ((preds == 0) & (labels == 0)).sum()
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fn + fp + epsilon)

    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return f1.item(), iou.item(), precision.item(), recall.item(), acc.item()


def check_accuracy(loader, model, writer, n_iter, device="cuda"):

    model_name = model.__class__.__name__
    # get loader name
    loader_name = loader.dataset.name

    f1_vals = 0.0
    iou_vals = 0.0
    precision_vals = 0.0
    recall_vals = 0.0
    acc_vals = 0.0

    model.eval()
    num_batches = len(loader)
    print("Checking accuracy...")
    with torch.no_grad():  # no need to calculate gradients, it is not training. Saves memory
        for x, y in loader:

            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (
                preds > 0.5
            ).float()  # binarize the predictions, works for multi-class as well

            f1, iou, precision, recall, acc = f1_and_iou(preds, y)
            f1_vals += f1  # =dice for binary classification
            iou_vals += iou
            precision_vals += precision
            recall_vals += recall
            acc_vals += acc

    writer.add_scalar(f"{model_name}/{loader_name}/f1", f1_vals / num_batches, n_iter)
    writer.add_scalar(f"{model_name}/{loader_name}/iou", iou_vals / num_batches, n_iter)
    writer.add_scalar(
        f"{model_name}/{loader_name}/precision", precision_vals / num_batches, n_iter
    )
    writer.add_scalar(
        f"{model_name}/{loader_name}/recall", recall_vals / num_batches, n_iter
    )
    writer.add_scalar(
        f"{model_name}/{loader_name}/acc1", acc_vals / num_batches, n_iter
    )
    writer.flush()

    model.train()

    return f1_vals / num_batches

# saves and plots all
# TODO remove
def save_predictions_as_imgs(
    loader,
    model,
    folder="/content/drive/MyDrive/Track1test/saved_images",
    device="cuda",
):
    model.eval()
    model_name = model.__class__.__name__
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        for id, (pred, gt) in enumerate(zip(preds, y)):
            img_idx = idx * len(loader) + id
            torchvision.utils.save_image(
                preds, f"{folder}/{model_name}_pred_{img_idx}.png"
            )

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Display the prediction
            ax[0].imshow(pred.squeeze().cpu().numpy(), cmap="gray")
            ax[0].set_title(f"Prediction {img_idx}")
            ax[0].axis("off")

            # Display the ground truth
            ax[1].imshow(gt.squeeze().cpu().numpy(), cmap="gray")
            ax[1].set_title(f"Ground Truth {img_idx}")
            ax[1].axis("off")

            plt.tight_layout()
            plt.savefig(f"{folder}/{model_name}_comparison_{img_idx}.png")
            if id == 0:
                display(fig)
            plt.close(fig)

            plt.show()

    model.train()


def plot_predictions(
    loader,
    model,
    folder="/content/drive/MyDrive/Track1test/saved_images",
    device="cuda",
):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        for id, (pred, gt) in enumerate(zip(preds, y)):
            img_idx = idx * len(loader) + id
            torchvision.utils.save_image(preds, f"{folder}/pred_{img_idx}.png")

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Display the prediction
            ax[0].imshow(pred.squeeze().cpu().numpy(), cmap="gray")
            ax[0].set_title(f"Prediction {img_idx}")
            ax[0].axis("off")

            # Display the ground truth
            ax[1].imshow(gt.squeeze().cpu().numpy(), cmap="gray")
            ax[1].set_title(f"Ground Truth {img_idx}")
            ax[1].axis("off")

            plt.show()

    model.train()



if __name__ == "__main__":

    # test the functions
    preds = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 1, 1, 0])

    f1, iou, precision, recall, acc = f1_and_iou(preds, labels)
    print(f"F1: {f1}, IoU: {iou}, Precision: {precision}, Recall: {recall}, Acc: {acc}")
    print(f1.type(), iou.type(), precision.type(), recall.type(), acc.type())
    print(f1.item(), iou.item(), precision.item(), recall.item(), acc.item())
    print(f1.shape, iou.shape, precision.shape, recall.shape, acc.shape)
    print(f1.dim(), iou.dim(), precision.dim(), recall.dim(), acc.dim())
