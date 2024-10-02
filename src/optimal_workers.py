# before this test download_data is called
# use that data(not the dummy data) to test the dataloader, simply iterate over it and measure the time
# put it in a seperate script and send this as the first job:
#    1. download_data
#    2. test_dataloader
#    3. write_results on json file like the one below on line 160

import time
import multiprocessing as mp
import psutil
import json
import torch
import os



from dataset import get_loaders
from config import LocalConfig, LocalTestConfig, ClusterConfig, ClusterTestConfig, CurrentInstance,create_dirs
from download_data import download_data

SMALL_DATASET = CurrentInstance.SMALL_DATASET
PIN_MEMORY = True if torch.cuda.is_available() else False

def get_active_config():
    if torch.cuda.is_available():
        return ClusterTestConfig if SMALL_DATASET else ClusterConfig
    else:
        return LocalTestConfig if SMALL_DATASET else LocalConfig

ActiveConfig = get_active_config()


# Use the configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_HEIGHT = ActiveConfig.IMAGE_HEIGHT
IMAGE_WIDTH = ActiveConfig.IMAGE_WIDTH

TRAIN_IMG_DIR = ActiveConfig.TRAIN_IMG_DIR
TRAIN_MASK_DIR = ActiveConfig.TRAIN_MASK_DIR
VAL_IMG_DIR = ActiveConfig.VAL_IMG_DIR
VAL_MASK_DIR = ActiveConfig.VAL_MASK_DIR

# CURRENT VAL SET DON'T HAVE MASKS SO THIS DELETES THE IMGS AND MOVES SOME OF THE TRAINING DATA TO VAL
def split_test_train_val(ratio=0.2):

    img_files = os.listdir(TRAIN_IMG_DIR)
    mask_files = os.listdir(TRAIN_MASK_DIR)

    # order the files
    img_files.sort()
    mask_files.sort()

    print("img_files", img_files)
    print(len(img_files))
    print("mask_files", mask_files)
    print(len(mask_files))

    assert len(img_files) == len(mask_files), f"Number of images and masks do not match. {len(img_files)} != {len(mask_files)}"

    num_val_files = int(ratio * len(img_files))

    #exit(0)
    for file in os.listdir(VAL_IMG_DIR):
        os.remove(os.path.join(VAL_IMG_DIR, file))

    # move the files
    for i in range(num_val_files):
        os.system(f"mv {os.path.join(TRAIN_IMG_DIR, img_files[i])} {os.path.join(VAL_IMG_DIR, img_files[i])}")
        os.system(f"mv {os.path.join(TRAIN_MASK_DIR, mask_files[i])} {os.path.join(VAL_MASK_DIR, mask_files[i])}")

    # check if the files are moved
    assert len(os.listdir(VAL_IMG_DIR)) == num_val_files, f"Files are not moved to validation directory{VAL_IMG_DIR}. {len(os.listdir(VAL_IMG_DIR))} != {num_val_files}"
    assert len(os.listdir(VAL_MASK_DIR)) == num_val_files, f"Files are not moved to validation directory{VAL_MASK_DIR}. {len(os.listdir(VAL_MASK_DIR))} != {num_val_files}" 

    print("Files moved to validation directory")



def test_dataloader(num_workers, batch_size=32):
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

    start_time = time.time()
    for _ in train_loader:
        pass

    for _ in val_loader:
        pass
    end_time = time.time()

    return end_time - start_time


def find_optimal_workers(max_workers=None):
    if max_workers is None:
        max_workers = mp.cpu_count()

    results = {}
    for num_workers in range(0, max_workers + 1):
        elapsed_time = test_dataloader(num_workers)
        results[num_workers] = elapsed_time
        print(f"Num workers: {num_workers}, Time: {elapsed_time:.2f} seconds")

        # Check CPU usage, do not use psutil
        cpu_percent = psutil.cpu_percent()
        print(f"CPU Usage: {cpu_percent}%")

        # If CPU usage is very high, we might have reached the limit
        if cpu_percent > 95:
            print("CPU usage very high. Stopping the search.")
            break

        # If adding more workers doesn't improve time significantly, stop
        if num_workers > 0 and results[num_workers] > results[num_workers - 1] * 0.95:
            print(
                "Adding more workers doesn't improve time significantly. Stopping the search."
            )
            break

    optimal_workers = min(results, key=results.get)
    print(f"\nOptimal number of workers: {optimal_workers}")
    return optimal_workers


if __name__ == "__main__":
    # Download data
    #download_data()
    #create_dirs(LocalConfig, after_downloads=True)
    #split_test_train_val()


    # Find optimal number of workers
    optimal_workers = find_optimal_workers()
    print(f"Optimal number of workers: {optimal_workers}")

    # Write results to a JSON file
    results = {"optimal_workers": optimal_workers}
    with open("results.json", "w") as f:
        json.dump(results, f)
