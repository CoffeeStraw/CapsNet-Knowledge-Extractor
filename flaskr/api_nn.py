"""
Core for the backend of the project.
All the computations on the Neural Networks are done here.
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import PIL.Image as pil
import numpy as np

import matplotlib

# Set non interactive backend for matplolib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project's imports
from flaskr import paths

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model

# Utilities
from utils import pickle_load
from capslayers import compute_vectors_length


def load_model(model_name):
    """
    """
    # Import model (HACK)
    sys.path.append(os.path.join(paths["trainer"], model_name))
    from capsnet import CapsuleNet

    sys.path.pop()

    # Load model
    try:
        model_params = pickle_load(
            os.path.join(paths["data"], model_name, "model_params.pkl")
        )
    except FileNotFoundError:
        return None

    model_params["batch_size"] = 1
    model_params.pop("dataset")
    _, model = CapsuleNet(**model_params)

    # Clean modules (HACK)
    del sys.modules["capsnet"]

    return model, model_params


def get_processable_layers(layers):
    """
    Returns computable layers from a list of layers, along with their types. 
    We check that by searching specific keywords in layer's name,
    since we can't be sure about layer's type.

    If you wish to extend the project and add compatibility
    for more layers, you should as well update this function.
    """
    processable_layers = []

    for layer in layers:
        layer_name = layer.name.lower()

        if "conv" in layer_name:
            processable_layers.append([layer_name, "CONVOLUTIONAL"])
        elif "primary" in layer_name and "caps" in layer_name:
            processable_layers.append([layer_name, "PRIMARY_CAPS"])
        elif "caps" in layer_name:
            processable_layers.append([layer_name, "CLASS_CAPS"])
        elif "mask" in layer_name:
            processable_layers.append([layer_name, "MASK"])

    return processable_layers


def conv_out_process(act, layer_img_dir):
    # Prepare new image to contain convolutional's features
    new_img_width = act.shape[0] * act.shape[-1]
    new_img_height = act.shape[1]
    new_img = pil.new("L", (new_img_width, new_img_height))

    # Save features
    act = tf.multiply(act, 255.0)
    for i in range(act.shape[-1]):
        feature = act[:, :, i].numpy()
        feature_image = pil.fromarray(feature).convert("L")

        new_img.paste(
            feature_image, (act.shape[0] * i, 0),
        )

    # Save filters (TODO?)
    pass

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    return {
        "rows": 1,
        "cols": act.shape[-1],
        "chunk_width": act.shape[0],
        "chunk_height": act.shape[1],
        "outs": "out.jpeg",
    }


def pcap_out_process(layer, config, act, layer_img_dir):
    # Get new features' dimension
    feature_dim = int(
        (layer.input_shape[1] - config["kernel_size"] + 1) / config["strides"]
    )

    # Compute vectors' length and reshape
    act = tf.reshape(compute_vectors_length(act), (feature_dim, feature_dim, -1))

    # Prepare new image to contain capsules' activations
    new_img_width = feature_dim * act.shape[-1]
    new_img_height = feature_dim
    new_img = pil.new("L", (new_img_width, new_img_height))

    # Save as images with the same size of the inputs
    act = tf.multiply(act, 255.0)
    for i in range(act.shape[-1]):
        capsules_length = act[:, :, i].numpy()
        pcaps_image = pil.fromarray(capsules_length).convert("L")

        new_img.paste(
            pcaps_image, (feature_dim * i, 0),
        )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    return {
        "rows": 1,
        "cols": act.shape[-1],
        "chunk_width": feature_dim,
        "chunk_height": feature_dim,
        "outs": "out.jpeg",
    }


def ccap_out_process():
    pass


def mask_out_process(layer, act, layer_img_dir, model_params):
    # Get next layer ( suppose decoder (HACK) )
    next_layer = layer._outbound_nodes[1].outbound_layer

    # Prepare variables for the iterations
    act_numpy = act.numpy()
    start = np.argmax(act != 0)
    n_dims = act.shape[0] // model_params["n_class"]
    n_manip = 11

    # Prepare new output image
    new_img_width = model_params["input_shape"][0] * n_manip
    new_img_height = model_params["input_shape"][1] * n_dims
    new_img = pil.new("L", (new_img_width, new_img_height))

    # Edit tensor values and feed the activation forward to get reconstruction
    for i_dim in range(start, start + n_dims):
        for i_r, r in enumerate(np.linspace(-0.25, 0.25, n_manip)):
            r = round(r, 2)
            # Edit dimension value
            act_to_feed = np.copy(act_numpy)
            act_to_feed[i_dim] += r

            # Convert back to tensor and feed-forward
            act_to_feed = tf.expand_dims(tf.convert_to_tensor(act_to_feed), 0)
            reconstructed_image = next_layer(act_to_feed)

            # Pre-process reconstruction
            reconstructed_image = tf.squeeze(reconstructed_image)
            reconstructed_image = tf.multiply(reconstructed_image, 255.0)
            reconstructed_image = reconstructed_image.numpy()

            # Convert to image and paste to the new image
            reconstructed_image = pil.fromarray(reconstructed_image).convert("L")
            new_img.paste(
                reconstructed_image,
                (
                    model_params["input_shape"][0] * i_r,
                    model_params["input_shape"][1] * (i_dim - start),
                ),
            )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    return {
        "rows": n_dims,
        "cols": n_manip,
        "chunk_width": model_params["input_shape"][0],
        "chunk_height": model_params["input_shape"][1],
        "outs": "out.jpeg",
    }


def model_out_process(predictions, model_out_dir):
    """
    Visualize model's predictions in an histogram.
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()

    classes = np.arange(len(predictions))

    ax.barh(classes, predictions, align="center")
    ax.set_yticks(classes)
    ax.set_yticklabels(classes)
    ax.set_xlim(0, 1.0)

    plt.title(
        f"Prediction: {np.argmax(predictions)}",
        fontsize=30.0,
        color="blue",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", colors="blue", labelsize=20.0)

    for i, v in enumerate(predictions):
        ax.text(
            v + 0.01,
            i + 0.06,
            str(round(v, 4)),
            color="blue",
            fontweight="bold",
            verticalalignment="center",
            size=15.0,
        )

    plt.savefig(os.path.join(model_out_dir, "out.png"), transparent=True)
    return {"outs": "out.png"}


def compute_step(model, model_params, prep_img, req_out_dir):
    """
    Computes and saves the outputs of the processable layers in the given model.
    """
    # Create a temporary model to retrieve all the layers' activations
    processable_layers = get_processable_layers(model.layers)
    processable_layers = [(model.get_layer(pl[0]), pl[1]) for pl in processable_layers]

    activation_model = Model(model.input, [pl[0].output for pl in processable_layers])
    activations = activation_model(prep_img)

    # Prepare dictionary to contain outputs
    out_dir = req_out_dir.replace("\\", "/")
    out_dir = out_dir[out_dir.find("/static") :] + "/"

    out_info = {"out_dir": out_dir}
    outs = {}

    for (layer, layer_type), act in zip(processable_layers, activations):
        # Get layer config
        config = layer.get_config()
        # Extract from batch
        act = act[0]

        # Prepare directory for the layer
        layer_name = config["name"]
        layer_img_dir = os.path.join(req_out_dir, layer_name)
        os.mkdir(layer_img_dir)

        # === PROCESS LAYER ACTIVATION BASED ON TYPE ===
        if layer_type == "CONVOLUTIONAL":
            outs[layer_name] = conv_out_process(act, layer_img_dir)
        elif layer_type == "PRIMARY_CAPS":
            outs[layer_name] = pcap_out_process(layer, config, act, layer_img_dir)
        elif layer_type == "CLASS_CAPS":
            outs[layer_name] = ccap_out_process()
        elif layer_type == "MASK":
            outs[layer_name] = mask_out_process(layer, act, layer_img_dir, model_params)
        else:
            raise TypeError(f"Layer type '{layer_type}' cannot be computed.")

    # Add layers outputs
    out_info["layers_outs"] = outs

    # Process model's output
    model_out_dir = os.path.join(req_out_dir, "model_out")
    os.mkdir(model_out_dir)

    predictions = model(prep_img)[0][0].numpy()

    out_info["model_out"] = model_out_process(predictions, model_out_dir)

    return out_info
