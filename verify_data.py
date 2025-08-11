import os
import requests
from tqdm import tqdm

# Dictionary mapping filenames to their unique download IDs
'''DATASET_FILES = {
    "espargos-0007-meanders-nw-se-1.tfrecords": "378515",
    "espargos-0007-meanders-sw-ne-1.tfrecords": "378516",
    "espargos-0007-randomwalk-1.tfrecords": "378514",
    "espargos-0007-randomwalk-2.tfrecords": "378518",
    "espargos-0007-human-helmet-randomwalk-1.tfrecords": "378523",
    "spec.json": "378510"
}'''

# All datasets from espargos-0007
DATASET_FILES = {
    "espargos-0007-meanders-nw-se-1.tfrecords": "378515",
    "espargos-0007-meanders-sw-ne-1.tfrecords": "378516",
    "espargos-0007-randomwalk-1.tfrecords": "378514",
    "espargos-0007-randomwalk-2.tfrecords": "378518",
    "espargos-0007-human-helmet-randomwalk-1.tfrecords": "378523",
    "spec.json": "378510",
    "espargos-0007-empty-room.tfrecords": "378511",
    "espargos-0007-spiral-ccw-1-part1.tfrecords": "378512",
    "espargos-0007-spiral-ccw-1-part2.tfrecords": "378513",
    "espargos-0007-radial-1.tfrecords": "378517",
    "espargos-0007-meanders-e-w-1.tfrecords": "378519",
    "espargos-0007-spiral-ccw-2.tfrecords": "378520",
    "espargos-0007-human-helmet-meanders-nw-se-1.tfrecords": "378521",
    "espargos-0007-human-helmet-meanders-sw-ne-1.tfrecords": "378522",
    "espargos-0007-human-helmet-standing-center-1.tfrecords": "378524",
    "espargos-0007-human-helmet-circle-1.tfrecords": "378525",
    "espargos-0007-circle-1.tfrecords": "378526",
    
}

BASE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/"

def download_files(files_to_download, target_dir):
    """
    Downloads a list of files from the data repository with a progress bar.
    """
    print(f"Creating directory at: {os.path.abspath(target_dir)}")
    os.makedirs(target_dir, exist_ok=True)

    for filename in files_to_download:
        file_id = DATASET_FILES[filename]
        url = f"{BASE_URL}{file_id}"
        dest_path = os.path.join(target_dir, filename)

        print(f"\nDownloading '{filename}'...")
        try:
            # Download with streaming to handle large files and show progress
            response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
            response.raise_for_status() # Raise an error if the download fails (e.g., 404)

            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=filename) as pbar:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"'{filename}' saved successfully.")

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Failed to download '{filename}'. Reason: {e}")
            print("Please check your internet connection and try again.")


def check_and_download_data():
    """
    Verifies the existence of the dataset directory and files, and offers
    to download them if anything is missing.
    """
    # Get the dataset directory name from the espargos_0007 module
    dataset_dir = "espargos_0007"
    
    # --- Check if the dataset directory exists ---
    if not os.path.exists(dataset_dir):
        print(f"\nWarning: Dataset folder '{dataset_dir}' not found.")
        response = input("Do you want to create the folder and download all necessary files? (y/n): ").lower()
        if response == 'y':
            download_files(list(DATASET_FILES.keys()), dataset_dir)
        else:
            print("Operation cancelled. The script cannot continue without the data.")
            return
    else:
        # --- If the directory exists, check if the files are inside ---
        print(f"\nFolder '{dataset_dir}' found. Verifying files...")
        missing_files = []
        for filename in DATASET_FILES.keys():
            full_path = os.path.join(dataset_dir, filename)
            if not os.path.exists(full_path):
                missing_files.append(filename)

        if not missing_files:
            print("All necessary files already exist. No action needed. ✅")
        else:
            print("\nWarning: The following files were not found:")
            for file in missing_files:
                print(f"  - {file}")
            
            response = input("\nDo you want to download the missing files? (y/n): ").lower()
            if response == 'y':
                download_files(missing_files, dataset_dir)
            else:
                print("Operation cancelled. Some files are still missing.")

if __name__ == '__main__':
    check_and_download_data()
    print("\nVerification complete.")