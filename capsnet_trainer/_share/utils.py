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


def plot_log(file_paths):
    """
    Draw a plot of the accuracy and the loss from data collected during the training.

    Args:
        file_paths: List of paths to the .csv you want to plot.
    """
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import csv
    import os

    # Load data
    datas = []
    for file_path in file_paths:
        columns = defaultdict(list)
        columns["name"] = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

        with open(file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for (k, v) in row.items():
                    if k == "accuracy":
                        k = "vec_len_accuracy"
                    elif k == "val_accuracy":
                        k = "val_vec_len_accuracy"
                    elif k == "epoch":
                        v = str(int(v) + 1)

                    columns[k].append(round(float(v), 4))

        datas.append(columns)

    # Define what to plot
    to_plot = [
        {"title": "Validation Loss", "col_name": "val_loss", "ylabel": "Loss"},
        {
            "title": "Training Accuracy",
            "col_name": "vec_len_accuracy",
            "ylabel": "Accuracy",
        },
        {
            "title": "Validation Accuracy",
            "col_name": "val_vec_len_accuracy",
            "ylabel": "Accuracy",
        },
    ]

    # Plot
    for plot_info in to_plot:
        for columns in datas:
            plt.plot(
                columns["epoch"], columns[plot_info["col_name"]], label=columns["name"]
            )

        plt.legend()
        plt.xticks(np.arange(min(columns["epoch"]), max(columns["epoch"]) + 1, 1.0))
        plt.xlabel("Epochs")
        plt.ylabel(plot_info["ylabel"])
        plt.title(plot_info["title"])
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
