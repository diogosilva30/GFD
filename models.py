"""
Module containing the DL models
"""
from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model, losses
from tensorflow.keras import optimizers, losses, metrics

import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


class BaseModel(ABC):
    """
    Abstract model
    """

    def __str__(self) -> str:
        """
        String representation of the model.

        Returns
        -------
        str
            The name of the class without `Model`.
        """
        return self.__class__.__name__.split("Model")[0]

    @abstractmethod
    def build(self, input_shape, **kwargs):
        """
        Concrete classes should implement this method to build
        a model instance, and return the uncompiled model.
        """

    def cross_validate(self, train_data_list, **build_kwargs):
        """
        Performs cross-validation on the model.

        Parameters
        ----------
        train_data_list: list
            A list where each element contains multiple images.
            (Each element corresponds to a fold)
        Returns
        -------
        Model
            A trained model
        """
        print(f"Cross validating {str(self)}")
        # Create lists for holding the metric values
        acc_per_fold = []  # accuracy
        auc_per_fold = []  # AUC
        loss_per_fold = []  # loss

        # Create 2 empty lists for holding the training history losses and test history losses
        fold_train_loss, fold_test_loss = list(), list()

        # Iterate an index to size of folds
        for i in range(len(train_data_list)):
            # Extract validation data
            x_val, y_val = train_data_list[i]

            # Indices for train
            indices_to_keep = np.delete(range(len(train_data_list)), i)

            # Get train data
            x_train, y_train = _build_train_data(train_data_list, indices_to_keep)

            # Compute class weights for balanced learning. Returns a list
            weights = compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )

            # Convert the list do dict.
            weights_dict = {idx: value for idx, value in enumerate(weights)}

            # Build the model
            baseline_model = self.build(
                input_shape=x_train[0].shape,
                **build_kwargs,
            )
            # Compile the model
            baseline_model.compile(
                optimizer=optimizers.Adam(),
                loss=losses.BinaryCrossentropy(),
                metrics=[metrics.BinaryAccuracy(), metrics.AUC(name="auc")],
            )

            # Train the model with the training indices
            history = baseline_model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                batch_size=64,
                epochs=40,
                class_weight=weights_dict,
                verbose=0,
            )

            fold_train_loss.append(history.history["loss"])
            fold_test_loss.append(history.history["val_loss"])
            # Evaluate the model with the testing indices
            scores = baseline_model.evaluate(x_val, y_val, verbose=0)

            # Scores has format : ['loss', 'binary_accuracy', 'auc']
            # Save the loss value on the val data
            loss_per_fold.append(scores[0])
            # Save the Accuracy value on the val data
            acc_per_fold.append(scores[1])
            # Save the Accuracy value on the val data
            auc_per_fold.append(scores[2])

        # Plot the loss curve
        self._plot_losses(
            fold_train_loss,
            fold_test_loss,
        )

        # Display the results
        print(f"Accuracy: {acc_per_fold.mean():.2f} (+/- {acc_per_fold.std():.2f})")
        print(f"AUC: {auc_per_fold.mean():.2f} (+/- {auc_per_fold.std():.2f})")

        return baseline_model

    def _plot_losses(self, train_losses: list, test_losses: list):
        """
        Plots the losses
        """
        train_loss_mean = np.mean(train_losses, axis=0)
        train_loss_std = np.std(train_losses, axis=0)

        test_loss_mean = np.mean(test_losses, axis=0)
        test_loss_std = np.std(test_losses, axis=0)

        # Get number of epochs
        epochs = list(range(1, len(train_loss_mean) + 1))

        plt.fill_between(
            epochs,
            train_loss_mean - train_loss_std,
            train_loss_mean + train_loss_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            epochs,
            test_loss_mean - test_loss_std,
            test_loss_mean + test_loss_std,
            alpha=0.1,
            color="g",
        )

        plt.plot(epochs, train_loss_mean, "o-", color="r", label="Training loss")
        plt.plot(epochs, test_loss_mean, "o-", color="g", label="Test loss")
        plt.legend(loc="best")
        plt.xlabel("Epochs")
        plt.ylabel("Binary Crossentropy Loss")
        plt.title(str(self))
        plt.gca().set_ylim(bottom=0)
        plt.show()


class BaselineModel(BaseModel):
    def build(self, input_shape, **kwargs):
        """
        Builds a baseline model.
        Required kwargs:
            - `fully_connected_size`: Number of neurons in the FC layers.
            - `add_dropout_layers`: Whether to use Dropout layers.
        """

        input_img = layers.Input(shape=input_shape)

        # Conv 1
        x = layers.Conv2D(32, 3)(input_img)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv 2
        x = layers.Conv2D(64, 3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D(2)(x)

        # Conv 3
        x = layers.Conv2D(64, 3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Flatten()(x)

        # Fully-connected 1
        x = layers.Dense(kwargs["fully_connected_size"], activation="relu")(x)

        if kwargs["add_dropout_layers"]:
            x = layers.Dropout(0.5)(x)

        # Fully-connected 2
        x = layers.Dense(kwargs["fully_connected_size"], activation="relu")(x)

        if kwargs["add_dropout_layers"]:
            x = layers.Dropout(0.5)(x)

        # Output layer
        output = layers.Dense(1, activation="sigmoid")(x)

        # Build model
        model = Model(input_img, output)

        return model


class FullyConnectedEncoderModel(BaseModel):
    def build(self, input_shape, **kwargs):
        """
        Builds and returns a fully connected encoder
        """
        # Extract the encoder
        encoder = kwargs["autoencoder"].encoder

        # Freeze the encoder
        encoder.trainable = False

        # Input layer
        inputs = layers.Input(shape=input_shape)

        x = encoder(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.5)(x)

        # Output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)
        # Build model
        model = tf.keras.Model(inputs, outputs)
        return model


class Autoencoder(Model):
    def __init__(self, input_shape, n_bands: int):
        super().__init__()
        # Define encoder
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                ),
                layers.LeakyReLU(0.2),
                layers.Conv2D(24, (3, 3), padding="same"),
                layers.LeakyReLU(0.2),
            ]
        )

        # Define decoder (Provide input_shape for automatic model build)
        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(
                    24,
                    kernel_size=3,
                    padding="same",
                    input_shape=self.encoder.layers[-1].output_shape[1:],
                ),
                layers.LeakyReLU(0.2),
                layers.Conv2DTranspose(
                    32,
                    kernel_size=3,
                    padding="same",
                ),
                layers.LeakyReLU(0.2),
                layers.Conv2D(
                    n_bands, kernel_size=(3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
