"""
File containing functions not concerning directly Neural Networks.
Author: Antonio Strippoli
"""
import pickle
from tensorflow.keras.utils import to_categorical
import numpy as np


def load_dataset(name="MNIST"):
    """Loads and prepares the wanted dataset. You can choose between:
        - MNIST
        - Fashion_MNIST
        - CIFAR10
    """
    if name == "MNIST":
        from tensorflow.keras.datasets import mnist

        data = mnist.load_data()
    elif name == "Fashion_MNIST":
        from tensorflow.keras.datasets import fashion_mnist

        data = fashion_mnist.load_data()
    elif name == "CIFAR10":
        from tensorflow.keras.datasets import cifar10

        data = cifar10.load_data()
    else:
        raise ValueError(f"Given dataset name ({name}) is not implemented.")

    # Unpack data
    (x_train, y_train), (x_test, y_test) = data

    # Preprocess data
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train.astype("float32"))
    y_test = to_categorical(y_test.astype("float32"))

    return (x_train, y_train), (x_test, y_test)


def plot_log(file_path):
    """
    Draw a plot of the accuracy and the loss from data collected during the training.

    Args:
        file_path: Path to the .csv you want to plot.
    """
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import csv

    # Load data
    columns = defaultdict(list)
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(round(float(v), 4))

    # Plot loss
    plt.plot(columns["epoch"], columns["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.show()

    # Plot accuracy
    plt.plot(
        columns["epoch"], columns["vec_len_accuracy"], "g", label="vec_len_accuracy",
    )
    plt.plot(
        columns["epoch"],
        columns["val_vec_len_accuracy"],
        "b",
        label="val_vec_len_accuracy",
    )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")

    # fig.savefig('log.png')
    plt.show()


def pickle_dump(obj, path):
    """
    Save a serializable object with pickle,
    to reduce verbosity in the main file.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    """
    Loads a serialized object with pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
