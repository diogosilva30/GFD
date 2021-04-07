"""
Module containing the method for splitting the dataset
into k-stratified folds.
"""
import os
import glob
import shutil
import numpy as np

from distutils.dir_util import copy_tree
from sklearn.model_selection import StratifiedKFold

from .common import load_labels


def stratified_split(
    dataset_folder: str,
    k: int,
):
    """
    Splits the dataset into k-folds (with one holdout fold)

    Parameters
    ----------
    dataset_folder: str
        The path to the cleaned dataset folder.
    k: int
        The number of folds to split. The last one is the holdout fold.
    """
    # Find cleaned dataset folders
    dataset_folders = glob.glob(dataset_folder + "/**")
    # Find labels for the folders
    labels = load_labels(dataset_folders)

    folds = []
    for _, test_folder_idx in StratifiedKFold(n_splits=6, shuffle=True).split(
        dataset_folders, labels
    ):
        folds.append(np.array(dataset_folders)[test_folder_idx])

    base_dir = "splitted_dataset"
    # Ensure that new directory is created
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)

    # Save the folds
    for k, fold in enumerate(folds):
        print(f"Saving fold {k+1}")
        dest_dir = base_dir + f"/fold_{k+1}/"
        os.mkdir(dest_dir)
        for file in fold:
            final_dest = dest_dir + file.split("\\")[1]
            os.mkdir(final_dest)
            copy_tree(file, final_dest)