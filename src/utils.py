import os
import torch
import matplotlib.pyplot as plt
import torchvision
from IPython.display import display


"""
Contents:
    - set_seed
    - save_checkpoint
    - load_checkpoint
    - transfer_weights_without_1stblock
    - save_all_predictions
    - plot_predictions
"""


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Dataloader randomly reseeds workers so you might have to manually set its worker_init_fn as well


def save_checkpoint(state, directory, model_name):
    os.makedirs(directory, exist_ok=True)
    filename = directory + "/" + model_name + "_checkpoint.pth.tar"
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, scaler):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])


# to use the learned weights with different input channels
def transfer_weights_without_1stblock(src_checkpoint, dest_model):
    src_state = src_checkpoint["state_dict"].state_dict()
    dest_state = dest_model.state_dict()

    for key in src_state.keys():
        if key == "encoder_joint.0" or "encoders1_0":
            continue
        elif key in dest_state.keys():
            if src_state[key].shape == dest_state[key].shape:
                dest_state[key] = src_state[key]
            else:
                print(f"Shapes of {key} do not match.")

    dest_model.load_state_dict(dest_state)


def save_all_predictions(loader, model, folder, device="cuda"):
    model.eval()  # Ensure the model is in evaluation mode
    batch_size = loader.batch_size
    total_processed = 0
    model_name = model.__class__.__name__
    os.makedirs(folder, exist_ok=True)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Process each image in the batch
        for i in range(preds.size(0)):
            img_idx = batch_idx * batch_size + i

            # Save prediction
            torchvision.utils.save_image(preds[i], f"{folder}/{model_name}_pred_{img_idx}.png")

            # Create comparison plot
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Display the prediction
            ax[0].imshow(preds[i].squeeze().cpu().numpy(), cmap="gray")
            ax[0].set_title(f"Prediction {img_idx}")
            ax[0].axis("off")

            # Display the ground truth
            ax[1].imshow(y[i].squeeze().cpu().numpy(), cmap="gray")
            ax[1].set_title(f"Ground Truth {img_idx}")
            ax[1].axis("off")

            # Title of the image: model name
            fig.suptitle(model_name)

            plt.tight_layout()
            plt.savefig(f"{folder}/{model_name}_comparison_{img_idx}.png")
            display(fig)
            plt.close(fig)

            total_processed += 1

    print(f"Processed {total_processed} images from the validation set.")


def save_predictions_as_imgs(
    loader,
    model,
    folder,
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
            img_idx = idx * len(loader) + id  # büyük ihtimalle yanlis
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
