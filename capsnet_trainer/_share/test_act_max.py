"""
Test of Activation Maximization on Capsule Networks, to test robustness of dimensions.
Usage: python test_act_max modelName
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import shutil

# Add wanted capsule network folder to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(os.path.dirname(curr_dir), sys.argv[1])
sys.path.append(model_dir)

# Import CapsuleNetwork model for MNIST classification
from capsnet import CapsuleNet

# Import general training method for Capsules
from training import train, test, margin_loss

# Import some utilities
from utils import load_dataset, pickle_dump

# Visualization with Keras-vis
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate
from tf_keras_vis.utils.regularizers import L2Norm, TotalVariation
from tf_keras_vis.utils.callbacks import Print
from tf_keras_vis.gradcam import GradcamPlusPlus as Gradcam
from tf_keras_vis.utils import normalize
from capslayers import compute_vectors_length
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

import PIL.Image as pil


def activation_maximization(model, output_layer_name, losses, seed_input):
    """
    Visualize internal representation using activation maximization
    """
    activation_maximization = ActivationMaximization(eval_model, clone=False)

    activation = activation_maximization(
        losses,
        seed_input=seed_input,
        steps=300,
        input_modifiers=[],
        regularizers=[],
        input_range=(0.0, 1.0),
        callbacks=[Print(interval=50)],
    )
    image = (activation[0, :, :, 0] * 255.0).astype(np.uint8)

    subplot_args = {
        "nrows": 1,
        "ncols": 1,
        "figsize": (5, 5),
        "subplot_kw": {"xticks": [], "yticks": []},
    }

    f, ax = plt.subplots(**subplot_args)
    ax.imshow(image, cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Check existance of directories and delete them
    # (do not change the names or visualizer will not recognize them anymore)
    save_dir = os.path.join(model_dir, "outs")
    weights_save_dir = os.path.join(save_dir, "weights")

    # Load dataset
    dataset = "MNIST"
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

    # Set model args
    model_params = {
        "input_shape": x_train.shape[1:],
        "n_class": y_train.shape[1],
        "name": os.path.basename(os.path.dirname(__file__)),
    }

    # Instantiate Capsule Network Model
    model, eval_model = CapsuleNet(**model_params)
    eval_model.load_weights(os.path.join(weights_save_dir, "trained.h5"))

    # ACTIVATION MAXIMIZATION
    def model_modifier(current_model):
        target_layer = current_model.get_layer(name="digit_caps")
        new_model = tf.keras.Model(
            inputs=current_model.inputs, outputs=target_layer.output
        )
        return new_model

    eval_model = model_modifier(eval_model)
    target_input = tf.expand_dims(x_test[0], 0)
    target_encode = eval_model(target_input)[0][0]

    def loss(output):
        # Extract from batch
        output = output[0]
        # print(compute_vectors_length(output))
        loss = tf.keras.losses.MSE(output, target_encode)
        return -1 * loss

    seed_input = np.zeros((1, 28, 28, 1))
    activation_maximization(eval_model, "digit_caps", [loss, lambda x: 0], seed_input)
