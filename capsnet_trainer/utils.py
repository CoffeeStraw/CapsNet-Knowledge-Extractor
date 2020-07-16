"""
Utilities file, it contains functions not concerning Neural Networks.
It is usefull to separate logic.
Author: Antonio Strippoli
"""
import os
import argparse
import pickle


def load_mnist():
    """Loads and prepares MNIST dataset.
    """
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # Preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train.astype("float32"))
    y_test = to_categorical(y_test.astype("float32"))

    return (x_train, y_train), (x_test, y_test)


def parse_args():
    """
    Parses arguments from command line and returns them.
    """
    # Getting/setting the hyper-parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")

    # General
    parser.add_argument(
        "--save_dir",
        default="flaskr/data",
        help="The directory that will contains every output of the execution. Relative to project directory.",
    )
    parser.add_argument(
        "--save_freq",
        default=100,
        type=int,
        help="The number of batches after which weights are saved.",
    )

    # Testing or training?
    parser.add_argument(
        "-t",
        "--testing",
        action="store_true",
        help="Test the trained model on testing dataset",
    )

    # Initial weights?
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="The path of the saved weights. Should be specified when testing",
    )

    # Training
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs for the training."
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Size of the batch used for the training.",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument(
        "--lr_decay",
        default=0.9,
        type=float,
        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs",
    )

    # Capsule Network
    parser.add_argument(
        "--lam_recon",
        default=0.392,
        type=float,
        help="The coefficient for the loss of decoder",
    )
    parser.add_argument(
        "-r",
        "--routings",
        default=3,
        type=int,
        help="Number of iterations used in routing algorithm. should > 0",
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Construct save dir path for training weights
    args.weights_save_dir = os.path.join(args.save_dir, "weights")

    # Creating results directories, if they do not exist
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(args.save_dir):
        os.mkdir(os.path.join(project_dir, args.save_dir))
    if not os.path.exists(args.weights_save_dir):
        os.mkdir(os.path.join(project_dir, args.weights_save_dir))

    return args


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
