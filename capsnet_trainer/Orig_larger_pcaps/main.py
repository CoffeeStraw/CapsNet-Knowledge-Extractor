"""
CapsuleNetwork TF 2.2 Implementation
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
Author: Antonio Strippoli
"""
# General imports
import os
import sys

# Add _share folder to path, must do that before importing capsnet
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_share")
)

# Import CapsuleNetwork model for MNIST classification
from capsnet import CapsuleNet

# Import general training method for Capsules
from training import train, test

# Import some utilities
from utils import load_dataset, parse_args, pickle_dump


if __name__ == "__main__":
    # Set model name as parent folder name and set dataset name
    model_name = os.path.basename(os.path.dirname(__file__))
    dataset = "MNIST"

    # Parse args
    args = parse_args(model_name)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

    # Set model args and save them for later model reinstantiation
    model_params = {
        "input_shape": x_train.shape[1:],
        "n_class": y_train.shape[1],
        "dataset": dataset,
    }
    pickle_dump(model_params, os.path.join(args.save_dir, "model_params.pkl"))
    model_params.pop("dataset")

    # Instantiate Capsule Network Model
    model, eval_model = CapsuleNet(**model_params)

    # TEST: Load latest weights, test the model and quit
    # eval_model.load_weights(os.path.join(args.save_dir, "weights", "trained.h5"))
    # test(model=eval_model, data=(x_test, y_test), args=args)
    # quit()

    # Show a complete summary
    model.summary()

    # Train!
    model = train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
