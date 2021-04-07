"""
Module containing common utility methods.
"""
import os
import glob
import spectral
import numpy as np
import tensorflow as tf
import cv2
import shutil
import random
import itertools

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def normalize_tensor(nd_array: np.ndarray) -> np.ndarray:
    """
    Normalizes a tensor to range 0-1

    Parameters
    ----------
    nd_array: np.ndarray
        Numpy's tensor to normalize.

    Returns
    -------
    np.ndarray
        Normalized Numpy's tensor.
    """
    return (
        tf.divide(
            tf.subtract(nd_array, tf.reduce_min(nd_array)),
            tf.subtract(tf.reduce_max(nd_array), tf.reduce_min(nd_array)),
        )
    ).numpy()


def load_labels(x_files: list):
    """
    Loads the labels
    """
    # Lambda function to get label of folder name
    get_label = lambda folder_name: 0 if "NORMAL" in folder_name else 1
    # Extract label for each image
    return np.fromiter((get_label(img) for img in x_files), int)


def _get_input(
    path,
    bands_to_keep: list,
    resize: int = None,
):
    """
    Reads HSI from disk
    """
    folder_contents = sorted(glob.glob(path + "/**"))
    # print(path)
    hdr_file = list(filter(lambda file: ".hdr" in file, folder_contents))[0]
    # print(hdr_file)
    # .hdr image is at index 0. Resize image to common size. Keep only K Bands
    image = (
        spectral.open_image(hdr_file)
        .load()[:702, :640, bands_to_keep]
        .astype(np.float32)
    )
    if resize is None:
        return normalize_tensor(image)
    else:
        return normalize_tensor(cv2.resize(image, (0, 0), fx=resize, fy=resize))


def load_features(
    x_files,
    bands_to_keep: list,
    resize: int = None,
):
    """
    Loads the hyperspectral images
    """

    return np.array(
        [
            _get_input(
                img,
                bands_to_keep,
                resize,
            )
            for img in x_files
        ]
    )


def _patchify(
    image: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Splits a single image into N patches.

    Parameters
    ----------
    image: np.ndarray
        The image to patchify.
    patch_size: int
        The size of each patch.

    Returns
    -------
    np.ndarray
        Array containing the patches of the image.
    """
    # Save number of channels
    n_channels = image.shape[2]

    # Make image a batch
    image = tf.expand_dims(image, 0)

    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size * 0.80, patch_size * 0.80, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    # Perform reshape
    patches = tf.reshape(patches, (-1, patch_size, patch_size, n_channels))

    # Delete patches than are >=40 % black
    final_patches = []
    for patch in patches:
        # Calculate total number of pixels
        number_total_pixels = np.prod(patch.shape)
        # Calculate total number of black pixels (0)
        number_black_pixels = np.count_nonzero(patch == 0)
        # Calculate black pixels ration
        black_pixels_ratio = number_black_pixels / number_total_pixels
        # If patch is less than 40% black, keep it
        if black_pixels_ratio < 0.4:
            final_patches.append(patch)
    # Return patches as np array
    return np.array(final_patches)


def patchify_folds(
    folds_folders: list,
    destination: str,
    patch_size: int = 64,
):
    """
    Performs image patching on the folds.

    Parameters
    ----------
    folds_folders: list
        The image folders of the folds
    destination: str
        Destination path to save the patches.
    patch_size: int, optional, default = 64
        The size of each patch (`patch_size * patch_size`)
    """

    # If already exists, delete entire tree
    if os.path.exists(destination):
        # Force delete
        shutil.rmtree(destination, ignore_errors=True)

    for fold_folder in folds_folders:
        # Grab name of current fold
        current_fold = fold_folder.split("\\")[1]
        # Search for folder contents
        fold_images = glob.glob(fold_folder + "/*")

        # Iterate over each image folder present in the current fold
        for image_folder in fold_images:
            # Get file name
            file_name = image_folder.split("\\")[-1].split(".hdr")[0]
            print(f"File {file_name}:")
            # Search folder contents
            image_folder_contents = sorted(glob.glob(image_folder + "/*"))
            # Get the .hdr file
            image_file = list(
                filter(lambda file: ".hdr" in file, image_folder_contents)
            )[0]
            # Load the .hdr file
            image = spectral.open_image(image_file).load()
            # Generate patches
            image_patches = _patchify(image, patch_size)
            print(
                f"\tGenerated {len(image_patches)} patches of size ({patch_size}, {patch_size})"
            )
            for i, patch in enumerate(image_patches):
                # Generate new format name, ie: "IMAGE_1_PATCH_2_NORMAL"
                new_image_name = f"{file_name}_PATCH_{i+1}"
                # Generate path
                path = f"{destination}/{current_fold}/{new_image_name}/"
                # Create folders
                os.makedirs(path)
                # Create final path
                final_path = f"{path}{new_image_name}"
                # Save the patch
                spectral.envi.save_image(final_path + ".hdr", patch, dtype=np.float32)
                # Save rgb version
                spectral.save_rgb(final_path + ".jpg", patch, [29, 19, 9])
            print("**" * 30)


def load_fold(fold: str, bands_to_keep: list):
    """Balances, shuffles, loads and normalizes a fold"""
    # Search all image folders of this fold
    image_folders = glob.glob(fold + "/*")
    # Load labels for the images of this fold
    image_labels = load_labels(image_folders)

    # Grab indexs of positive labels
    positive_labels_indexs = np.argwhere(image_labels == 1)[:, 0]

    # Grab indexs of negative labels and select 'k' to match size of positive labels
    negative_labels_indexs = np.random.choice(
        np.argwhere(image_labels == 0)[:, 0], len(positive_labels_indexs)
    )

    # Concat the indexes
    total_indexs = np.concatenate(
        (positive_labels_indexs, negative_labels_indexs), axis=None
    )

    # Use the indexes to filter the image_folders
    selected_images = np.array(image_folders)[total_indexs]
    random.shuffle(selected_images)
    # Use the indexes to filter the image labels
    selected_images_labels = load_labels(selected_images)

    # Return a tupple of the normalized images and the respective labels
    return load_features(selected_images, bands_to_keep), selected_images_labels


def print_fold_stats(fold_folder=None, fold_labels=None):
    """Display fold stats by receiving the folder or the labels"""
    if fold_labels is None:
        # Get the folder images of this fold
        fold_images = sorted(glob.glob(fold_folder + "/*"))
        # load the labels for this images
        fold_labels = load_labels(fold_images)

    # Calculate number of positive labels
    positive_labels = np.sum(np.array(fold_labels) == 1)
    # Calculate number of negative labels
    negative_labels = np.sum(np.array(fold_labels) == 0)
    # Calculate total number of labels
    total_labels = len(fold_labels)

    print(f"\tNumber of images: {len(fold_labels)}")
    print(
        f"\tNumber of positive samples: {positive_labels} ({(positive_labels/(total_labels))*100:.2f} %)"
    )
    print(
        f"\tNumber of negative samples: {negative_labels} ({(negative_labels/(total_labels))*100:.2f} %)"
    )
    print("**" * 30)


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=True
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    plt.show()


def build_train_data(train_list, indices_to_keep) -> tuple:
    train = np.array(train_list, dtype=object)[indices_to_keep]
    x = []
    y = []
    for item in train:
        x.append(item[0])
        y.append(item[1])

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)
