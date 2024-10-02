import os
import sys
import traceback
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import tifffile as tiff

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

# TODO: implement self.labeled so that the dataset can be used for inference


"""
Contents:
    - SARDataset
            
            band1 = ds.GetRasterBand(1).ReadAsArray()  # SAR VV
            band2 = ds.GetRasterBand(2).ReadAsArray()  # SAR VH
            band3 = ds.GetRasterBand(3).ReadAsArray()  # Merit DEM
            band4 = ds.GetRasterBand(4).ReadAsArray()  # Copernicus DEM
            band5 = ds.GetRasterBand(5).ReadAsArray()  # ESA world cover map
            band6 = ds.GetRasterBand(6).ReadAsArray()  # Water occurrence probability
            
                    image = np.dstack((band1, band2, band3, band4, band5, band6))
            
    - get_transforms
    - get_loaders
"""


class SARDataset(Dataset):  # RETURNS NONE MASK WHEN UNLABELED

    def __init__(self, image_dir, mask_dir=None, name=None, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.name = name
        self.labeled = self.mask_dir != None 
        self.images = os.listdir(image_dir)
        self.band_count = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if self.labeled:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", ".png"))

        try:
            """
            ds = gdal.Open(img_path)
            ds = Image.open(img_path)
            if ds is None:
                print("Failed to open the image")
                sys.exit(1)
            print(gdal.Info(gdal.Open(mask_path)))
            print(gdal.Info(ds))
            """

            # image = np.dstack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)])
            image = tiff.imread(img_path)
            self.band_count = image.shape[2]

            mask = None
            if self.labeled:
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int16) # already in 0-1 range

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                if self.labeled:
                    mask = augmentations["mask"]

        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()
            # don't exit, just raise the exception
            raise e

        return image, mask


def get_transforms(image_height, image_width, mean=[0.0] * 6, std=[1.0] * 6):  # possible TODO: adjust augmentations with grid search
    train_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(  # TODO: should be data oriented
                mean=mean,
                std=std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    return train_transforms, val_transforms


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    image_height,
    image_width,
    num_workers,
    pin_memory=True,
):
    train_transform, val_transform = get_transforms(image_height, image_width)
    train_ds = SARDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        name="train",
        transform=train_transform,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SARDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        name="val",
        transform=val_transform,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":


    dataset = SARDataset(
        "data/train/images",
        "data/train/labels",
    )
    print(dataset[1][0].shape)
    print(dataset.band_count)
    # plot a mask
    plt.imshow(dataset[1][1])
    plt.show()

