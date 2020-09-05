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
from PIL import ImageFilter


def gradcam_visualize(model):
    """
    Visualization using GRADCAM
    """

    def model_modifier(current_model):
        target_layer = current_model.get_layer(name="vec_len")
        new_model = tf.keras.Model(
            inputs=current_model.inputs, outputs=target_layer.output
        )
        return new_model

    gradcam = Gradcam(model, model_modifier=model_modifier, clone=False)

    def loss(output):
        """
        Maximize the prediction with greater probability
        """
        i = np.argmax(output, 1)[0]
        return output[0, i]

    for i in range(1, 100):
        X = x_test[i]
        cam = gradcam(
            loss, X, penultimate_layer=model.get_layer(name="digit_caps")
        )  # model.layers number
        cam = normalize(cam)

        subplot_args = {
            "nrows": 1,
            "ncols": 1,
            "figsize": (5, 5),
            "subplot_kw": {"xticks": [], "yticks": []},
        }

        f, ax = plt.subplots(**subplot_args)
        heatmap = np.uint8(cm.jet(cam)[0, ..., :3] * 255)
        ax.imshow(X[..., 0], cmap="gray")
        ax.imshow(heatmap, cmap="jet", alpha=0.5)  # overlay
        plt.tight_layout()
        plt.show()


def activation_maximization(model, output_layer_name, losses, seed_input):
    """
    Visualize internal representation using activation maximization
    """
    activation_maximization = ActivationMaximization(eval_model, clone=False)

    def gf(x, radius=0):
        x = (x[0, ..., 0].numpy() * 255.0).astype(np.int8)
        x = pil.fromarray(x, "L")
        x = x.filter(ImageFilter.GaussianBlur(radius=radius))
        x = np.array(x, dtype=np.float32) / 255.0
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)
        return x

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
    ax.imshow(image)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Check existance of directories and delete them
    # (do not change the names or visualizer will not recognize them anymore)
    save_dir = os.path.join(curr_dir, "outs")
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

    # GRADCAM
    # gradcam_visualize(eval_model)
    # quit()

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
        print(compute_vectors_length(output))
        loss = tf.keras.losses.MSE(output, target_encode)
        return -1 * loss

    seed_input = np.zeros((1, 28, 28, 1))
    # seed_input = tf.expand_dims(x_test[0], 0)
    # seed_input = None

    # seed_input = tf.expand_dims(x_test[0], 0)
    activation_maximization(eval_model, "digit_caps", [loss, lambda x: 0], seed_input)
