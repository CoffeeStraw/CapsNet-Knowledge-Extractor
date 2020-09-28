"""
Capsule Network training processes implementation.
Author: Antonio Strippoli
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
"""
import os
import numpy as np

# Import TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks


def margin_loss(y_true, y_pred):
    """Local function for margin loss, Eq.(4).

    When y_true[i, :] contains more than one `1`, this loss should work too (not tested).

    Args:
        y_true: Correct labels one-hot encoded (batch_size, n_classes)
        y_pred: Output of the DigitCaps Layer (batch_size, n_classes)
    Returns:
        A scalar loss value.
    """
    # Calculate loss
    L = y_true * tf.square(tf.maximum(0.0, 0.9 - y_pred)) + 0.5 * (
        1 - y_true
    ) * tf.square(tf.maximum(0.0, y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


class WeightsSaver(callbacks.Callback):
    """
    Callback that saves weights after every `save_freq` batches at `save_dir` directory.
    """

    def __init__(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        if batch % self.save_freq == 0:
            # Save model current state for later visualization
            save_name = os.path.join(self.save_dir, f"{self.epoch}-{batch}.h5")
            self.model.save_weights(save_name)


def train(
    model,
    data,
    epochs=10,
    batch_size=100,
    lr=0.001,
    lr_decay_mul=0.9,
    lam_recon=0.392,
    save_dir=None,
    weights_save_dir=None,
    save_freq=100,
):
    """Train a given Capsule Network model.

    Args:
        model: The CapsuleNet model to train.
        data: The dataset that you want to train: ((x_train, y_train), (x_test, y_test)).
        epochs: Number of epochs for the training.
        batch_size: Size of the batch used for the training.
        lr: Initial learning rate value.
        lr_decay_mul: The value multiplied by lr at each epoch. Set a larger value for larger epochs.
        lam_recon: The coefficient for the loss of decoder (if present).
        save_dir: Directory that will contain the logs of the training. `None` if you don't want to save the logs.
        weights_save_dir: Directory that will contain the weights saved. `None` if you don't want to save the weights.
        save_freq: The number of batches after which weights are saved.
    Returns:
        The trained model.
    """
    # Unpack data
    (x_train, y_train), (x_test, y_test) = data

    # Understand if the model uses the decoder or not
    n_output = len(model.outputs)

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=[margin_loss, "mse"] if n_output == 2 else [margin_loss],
        loss_weights=[1.0, lam_recon] if n_output == 2 else [1.0],
        metrics=["accuracy"],
    )

    # Define a callback to reduce learning rate
    cbacks = [
        callbacks.LearningRateScheduler(
            schedule=lambda epoch: lr * (lr_decay_mul ** epoch)
        )
    ]

    # Define a callback to save training datas
    if save_dir:
        cbacks.append(callbacks.CSVLogger(os.path.join(save_dir, "training.csv")))

    # Define a callback to save weights during the training
    if weights_save_dir:
        cbacks.append(WeightsSaver(weights_save_dir, save_freq))

    # Simple training without data augmentation
    model.fit(
        x=(x_train, y_train) if n_output == 2 else x_train,
        y=(y_train, x_train) if n_output == 2 else y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=((x_test, y_test), (y_test, x_test))
        if n_output == 2
        else (x_test, y_test),
        callbacks=cbacks,
    )

    # Save final weights at the end of the training
    if weights_save_dir:
        model.save_weights(os.path.join(weights_save_dir, "trained.h5"))

    return model


def test(model, data):
    """
    Calculate accuracy of the model on the test set.
    """
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print(
        "Test acc:",
        np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0],
    )
