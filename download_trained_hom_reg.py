
"""
This script downloads the networks trained for the paper.
Note that the only part of the architecture that actually gets trained is the
homography regression.
"""

import os
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id="14rEDzPYbLThfIHzT5wlP5le8ILQObqmf",
                                    dest_path="./tmp", unzip=True)
os.remove("tmp")
os.makedirs("trained_networks", exist_ok=True)
shutil.move("trained_homography_regressions", "trained_networks/trained_homography_regressions")

