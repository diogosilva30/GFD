# Import dependencies
import os
import glob
import numpy as np

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

from utils.common import (
    patchify_folds,
    print_fold_stats,
    load_fold,
)
from utils.stratified_split import stratified_split
from utils.common import build_train_data

from data_cleaning import clean

from models import BaselineModel, Autoencoder

from arguments_parser import get_parser


def prepare_data():
    """
    Cleans and processes the dataset
    """
    # 1. Clean the dataset
    clean(dataset_folder="dataset", masks_folder="masks")

    # 2. Split the dataset into 6 stratified folds (1 holdout)
    stratified_split("cleaned_dataset", k=6)

    # Now we have a folder called 'splitted_dataset', which
    # contains all the folds
    # Search for the folds
    folds_folders = glob.glob("splitted_dataset/*")

    # Patchify the folds, using the last one for the holdout
    patchify_folds(folds_folders, destination="patched_dataset")


def main():

    # Parse arguments
    args = get_parser().parse_args()

    # Check for GPU
    is_gpu = len(tf.config.list_physical_devices("GPU")) > 0

    if not is_gpu and not args.allow_cpu:
        raise ValueError(
            "Cannot run the code on CPU. Please enable GPU support or pass the '--allow-cpu' flag."
        )

    # Check if dataset should be recreated
    if not os.path.exists("cleaned_dataset") or args.force_recreate:
        prepare_data()

    # ........................................................................
    # Let's select the train folds and the holdout fold
    # ........................................................................
    # Grab folds folders (fold_1, ..., fold_k)
    folds_folders = glob.glob("patched_dataset/**")

    # Randomly select 1 fold for being the holdout final test
    holdout_fold = np.random.choice(folds_folders, 1)[0]

    # Keep the rest for training and performing k-fold
    # Search for elements of `folds_folders` that are not in `holdout_fold`
    train_folds = np.setdiff1d(folds_folders, holdout_fold)

    print(f"Train folds: {train_folds}")
    print(f"Holdout fold: {holdout_fold}")

    for k, fold in enumerate(train_folds):
        # Print current fold
        print(f"Train Fold {k+1}:")
        print_fold_stats(fold)

    # Now for the holdout fold
    print("Holdout Fold:")
    print_fold_stats(holdout_fold)

    # Generate the bands to keep
    bands_to_keep = np.round(np.linspace(0, 272 - 1, args.bands)).astype(int)

    # Load test images and labels
    print("Loading holdout fold images")
    images, labels = load_fold(holdout_fold, bands_to_keep)
    test_data = (images, labels)
    # Creat list for holding the training datasets folds
    train_data_list = []
    for k, fold in enumerate(train_folds):
        print(f"Loading images of training fold {k+1}")
        # Load the normalized images to list
        images, labels = load_fold(fold, bands_to_keep)
        # Save images in dict with labels as value
        train_data_list.append([images, labels])

    # Check if a baseline model should be created
    if args.baseline:
        # ........................................................................
        # Let's now establish a baseline model
        # ........................................................................
        model = BaselineModel().cross_validate(
            train_data_list,
            fully_connected_size=128,
            add_dropout_layers=True,
        )

        # How does the model perform on unseen data?
        result = model.evaluate(test_data[0], test_data[1], batch_size=64)
        result = dict(zip(model.metrics_names, result))
        print(result)

    # ........................................................................
    # Let's now try to use the Autoencoder
    # ........................................................................
    # Get train data (We only care about the features now)
    x_train, _ = build_train_data(train_data_list, range(len(train_data_list)))
    x_test = test_data[0]

    print("Training an Autoencoder")
    # Instantiate autoencoder
    autoencoder = Autoencoder(x_train[0].shape, n_bands=args.bands)

    # Prepare callbacks
    # 1. Reduce learning rate when loss is on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=15, min_lr=9e-4, verbose=1
    )
    # 2. Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=20,
        mode="auto",
        verbose=0,
        restore_best_weights=True,
    )

    # Compile the autoencoder
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss="mse",
    )

    # Train the model
    history = {}
    history["autoencoder"] = autoencoder.fit(
        x_train,
        x_train,
        validation_split=0.2,
        batch_size=16,
        epochs=250,
        verbose=1,
        callbacks=[reduce_lr, early_stop],
    )
    # Plot the Autoencoder loss curve
    plotter = tfdocs.plots.HistoryPlotter(metric="loss", smoothing_std=10)
    plotter.plot(history)


if __name__ == "__main__":
    main()