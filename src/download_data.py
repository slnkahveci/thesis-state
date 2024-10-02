import os
import tqdm
import boto3 # AWS SDK for Python
import zipfile
from config import ClusterConfig, ClusterTestConfig, CurrentInstance


AWS_ACCESS_KEY = CurrentInstance.AWS_ACCESS_KEY
AWS_SECRET_KEY = CurrentInstance.AWS_SECRET_KEY

SMALL_DATASET = CurrentInstance.SMALL_DATASET
DATA_PATH = ClusterConfig.DATA_DIR if SMALL_DATASET else ClusterTestConfig.DATA_DIR


def check_and_download_data(session,local_data_path, name,s3_bucket, s3_key):
    s3 = session.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

    file_name= os.path.join(local_data_path , name+".zip")

    if not os.path.exists(local_data_path):
        os.makedirs(local_data_path)
        download_required = True
    else:
        # Check if local data is up-to-date --- THIS IS NOT NECESSARY
        try:
            s3_object = s3.head_object(Bucket=s3_bucket, Key=s3_key)
            s3_last_modified = s3_object["LastModified"]
            local_last_modified = os.path.getmtime(local_data_path)
            download_required = s3_last_modified > local_last_modified
        except:
            download_required = True

    if download_required:
        kwargs = {"Bucket": s3_bucket, "Key": s3_key}
        # version id is not needed

        object_size = s3.head_object(**kwargs)["ContentLength"]

        print("Downloading data from AWS...")
        with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=file_name) as pbar:
            s3.download_file(
                Bucket=s3_bucket,
                Key=s3_key,
                ExtraArgs=None,
                Filename=file_name,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        print("Download complete.")

        print("Unzipping the downloaded file...")
        # os.system(f"unzip -o {file_name} -d {local_data_path}")
        # unzip command is not available in the cluster, use python instead

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(local_data_path)
        print("Unzip complete.")
        os.remove(file_name)
    else:
        print("Local data is up-to-date. No download required.")


def download_data():
    session = boto3.Session()
    s3_bucket = "ieee-dataport" 

    s3_key_train = "competition/1137541/Track1.zip"
    s3_key_test = "competition/1137541/Test Set Track 1.zip"

    temp_name = "Track1"

    local_data_path_train = DATA_PATH + "/train1/"
    local_data_path_test = DATA_PATH + "/test/"

    check_and_download_data(session, local_data_path_train, temp_name, s3_bucket, s3_key_train)

    # move all of os.path.join(DATA_PATH ,"Track1"')'s  contents to DATA_PATH because train contains the train+val
    os.system(f"mv {os.path.join(local_data_path_train ,temp_name)}/* {DATA_PATH}")
    os.system(f"rm -r {os.path.join(local_data_path_train)}")

    check_and_download_data(session, local_data_path_test, temp_name, s3_bucket, s3_key_test)

if __name__ == "__main__":
    download_data()