"""
This script downloads the networks trained for the paper.
Note that the only part of the architecture that actually gets trained is the
homography regression.
"""

import os
import gdown
import shutil

url = "https://drive.google.com/uc?id=14rEDzPYbLThfIHzT5wlP5le8ILQObqmf"
tmp_archive_name = "tmp.zip"
gdown.download(url, tmp_archive_name, quiet=False)

shutil.unpack_archive(tmp_archive_name)
os.remove(tmp_archive_name)
os.makedirs("data", exist_ok=True)
shutil.move("trained_homography_regressions", "data/trained_homography_regressions")
