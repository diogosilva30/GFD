"""
Module containing methods to clean the dataset
"""
import os
import shutil
import glob

import numpy as np
import spectral
import cv2


def clean(dataset_folder, masks_folder):
    """
    Cleans the dataset.
    """
    # Grab the image folders dirs
    image_folders = sorted(
        glob.glob(
            dataset_folder + "/*",
        )
    )

    # Grab the mask dirs
    mask_dirs = glob.glob(
        masks_folder + "/**/mask.*",
        recursive=True,
    )

    # Mask dirs contains .psd files. Use list comprehension to remove them. Sort them by number (ascending).
    # This is important so mask order matches image folder order
    mask_dirs = list(filter(lambda path: ".psd" not in path, mask_dirs))
    mask_dirs = sorted(
        mask_dirs, key=lambda file: int(file.split("\\")[-2].split("_")[1])
    )

    base_dir = "cleaned_dataset/"
    # Ensure that new directory is created
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)

    # Lambda function to process mask (Make it binary)
    process_mask = lambda mask: np.where(mask > 0, 255, 0) / 255

    for i, folder in enumerate(image_folders):
        # Get the folder name
        current_folder_name = folder.split("\\")[-1]
        # Extract the label. 1 is 'pos' in folder name, 0 otherwise
        label = "INFECTED" if "pos" in current_folder_name else "NORMAL"
        # Build new folder name
        new_folder_name = f"{current_folder_name.split('_')[0]}_{label}"

        # If folder does not exist, create it
        os.makedirs(base_dir + new_folder_name)
        # Find the contents of the folder
        folder_files = glob.glob(folder + "/**")
        # Filter for the .hdr file
        image_file = list(filter(lambda file: ".hdr" in file, folder_files))[0]
        # Load the .hdr file
        image = spectral.open_image(image_file).load()
        # Load the mask and process it
        mask = process_mask(cv2.imread(mask_dirs[i], 0)).astype(np.int8)
        # Apply mask to image
        print(f"Image {current_folder_name}:")
        print(f"\tApplying binary mask ({mask_dirs[i]})")
        image = cv2.bitwise_and(image, image, mask=mask)
        # Get final path
        path = base_dir + new_folder_name
        # Save .hdr image in final path
        print(f"\tSaving raw version ({path})")
        spectral.envi.save_image(
            path + "/" + new_folder_name + ".hdr", image, dtype=np.float32
        )
        # Save .jpg version with 3 random bands
        print(f"\tSaving jpg version ({path})")
        spectral.save_rgb(path + "/" + new_folder_name + ".jpg", image, [29, 19, 9])
        print("**" * 50)
