
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id="1WjfQFS_13uvg_eyefNJWVIKbvy9tSPR-",
                                    dest_path="./tmp", unzip=True)
os.remove("tmp")

