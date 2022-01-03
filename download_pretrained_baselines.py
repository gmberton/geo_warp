
"""
This script downloads the pretrained models, and saves them in data/pretrained_baselines.
These models have been trained following the training procedure as described
in the NetVLAD paper https://arxiv.org/abs/1511.07247
"""

import os
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id="1WjfQFS_13uvg_eyefNJWVIKbvy9tSPR-",
                                    dest_path="./tmp", unzip=True)
os.remove("tmp")
os.makedirs("data", exist_ok=True)
shutil.move("pretrained_baselines", "data/pretrained_baselines")

