"""
CapsuleNetwork TF 2.2 Implementation
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import shutil

# Add _share folder to path, must do that before importing capsnet
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(curr_dir), "_share"))

# Import CapsuleNetwork model for MNIST classification
from capsnet import CapsuleNet

# Import general training method for Capsules
from training import train, test

# Import some utilities
from utils import load_dataset, pickle_dump


if __name__ == "__main__":
    # Check existance of directories and delete them
    # (do not change the names or visualizer will not recognize them anymore)
    save_dir = os.path.join(curr_dir, "outs")
    weights_save_dir = os.path.join(save_dir, "weights")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)
    os.mkdir(weights_save_dir)

    # Load dataset
    dataset = "CIFAR10"
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

    # Set model args
    model_params = {
        "input_shape": x_train.shape[1:],
        "n_class": y_train.shape[1],
        "name": os.path.basename(os.path.dirname(__file__)),
    }

    # Instantiate Capsule Network Model
    model, eval_model = CapsuleNet(**model_params)

    # Save model args for later reinstantiation
    model_params["dataset"] = dataset
    pickle_dump(model_params, os.path.join(save_dir, "model_params.pkl"))

    # Show a complete summary
    model.summary()

    # Train!
    model = train(
        model=model,
        data=((x_train, y_train), (x_test, y_test)),
        save_dir=save_dir,
        weights_save_dir=weights_save_dir,
    )
