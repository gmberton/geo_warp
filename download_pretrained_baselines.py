"""
This script downloads the pretrained models, and saves them in data/pretrained_baselines.
These models have been trained following the training procedure as described
in the NetVLAD paper https://arxiv.org/abs/1511.07247
"""

import os
import gdown
import shutil

url = "https://drive.google.com/uc?id=1WjfQFS_13uvg_eyefNJWVIKbvy9tSPR-"
tmp_archive_name = "tmp.zip"
gdown.download(url, tmp_archive_name, quiet=False)

shutil.unpack_archive(tmp_archive_name)
os.remove(tmp_archive_name)
os.makedirs("data", exist_ok=True)
shutil.move("pretrained_baselines", "data/pretrained_baselines")

