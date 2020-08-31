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
            os.path.join(paths["trainer"], model_name, "outs", "model_params.pkl")
        )
    except FileNotFoundError:
        return None, None

    dataset = model_params.pop("dataset")
    _, model = CapsuleNet(**model_params)
    model_params["dataset"] = dataset

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


def conv_out_process(act, img_mode, layer_img_dir):
    # Extract from batch
    act = act[0]

    # Prepare new image to contain convolutional's features
    new_img_width = act.shape[0] * act.shape[-1]
    new_img_height = act.shape[1]
    new_img = pil.new("L", (new_img_width, new_img_height))

    # Save features
    act = tf.multiply(act, 255.0)
    for i in range(act.shape[-1]):
        feature = act[:, :, i].numpy().astype("int8")
        feature_image = pil.fromarray(feature)

        new_img.paste(
            feature_image, (act.shape[0] * i, 0),
        )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    # Save filters (TODO?)
    pass

    return {
        "Activation after ReLU": {
            "filename": "out.jpeg",
            "rows": 1,
            "cols": act.shape[-1],
            "chunk_width": act.shape[0],
            "chunk_height": act.shape[1],
        }
    }


def pcap_out_process(act, prep_img, img_mode, layer, layer_conf, layer_img_dir):
    # Extract from batch
    act = act[0]
    prep_img = prep_img[0]

    # Move preprocessed image to RGB domain (if not)
    if img_mode != "RGB":
        prep_img = np.repeat(prep_img[:, :] * 255.0, 3, axis=2)
    else:
        prep_img *= 255.0
    # Add alpha channel and convert to image
    prep_img = np.dstack((prep_img, np.full(prep_img.shape[:-1], 255.0))).astype("int8")
    prep_img = pil.fromarray(np.uint8(prep_img))

    # Get new features' dimension
    feature_dim = int(
        (layer.input_shape[1] - layer_conf["kernel_size"] + 1) / layer_conf["strides"]
    )
    # Compute vectors' length and reshape
    act = tf.reshape(compute_vectors_length(act), (feature_dim, feature_dim, -1))

    # Prepare new image to contain capsules' activations
    chunk_width, chunk_height = prep_img.size
    new_img_width = chunk_width * act.shape[-1]
    new_img_height = chunk_height
    new_img = pil.new("RGB", (new_img_width, new_img_height))

    # Save as images with the same size of the inputs
    act = tf.multiply(act, 255)
    for i in range(act.shape[-1]):
        capsules_length = act[:, :, i].numpy()
        capsules_length = np.interp(
            capsules_length, (capsules_length.min(), capsules_length.max()), (0.0, 1.0),
        )
        heatmap = pil.fromarray(np.uint8(matplotlib.cm.jet(capsules_length) * 255))

        # If we stop here, we can get a visualized matrix showing capsules activations,
        # but we go further, rescaling and superimposing the heatmap to the original image
        heatmap = heatmap.resize((chunk_width, chunk_height))
        pcaps_img = pil.blend(prep_img, heatmap, alpha=0.7)

        new_img.paste(
            pcaps_img, (chunk_width * i, 0),
        )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    return {
        "Capsules' length as heatmap": {
            "filename": "out.jpeg",
            "rows": 1,
            "cols": act.shape[-1],
            "chunk_width": chunk_width,
            "chunk_height": chunk_height,
        }
    }


def ccap_out_process(prev_layer_pack, layer_pack, prep_img, img_mode, layer_img_dir):
    # --------------------------
    # The following is a compact version of routing path visualization. All the credits goes to Aman Bhullar:
    # https://atrium.lib.uoguelph.ca/xmlui/bitstream/handle/10214/17834/Bhullar_Aman_202003_Msc.pdf?sequence=1&isAllowed=y
    # --------------------------

    # TODO: Implement the complete routing path visualization.
    # Also, atm we assume that the previous layer is always a primary caps, which is not always be the case

    # Unpack variables
    ((pl, _), prev_act) = prev_layer_pack
    ((cl, _), curr_act) = layer_pack

    pl_conf = pl.get_config()
    cl_conf = cl.get_config()

    # Get new features' dimension
    feature_dim = int(
        (pl.input_shape[1] - pl_conf["kernel_size"] + 1) / pl_conf["strides"]
    )

    # HACK: Assume that every input image to the network is squared,
    # which would produce squared features as well
    dims = [
        feature_dim,
        feature_dim,
        pl_conf["n_caps"],
        cl_conf["n_caps"],
    ]

    # Prepare output of previous caps layer
    tmp_dims = dims[0:3] + [1] * (len(dims) - 3)
    prev_caps_lengths = tf.reshape(compute_vectors_length(prev_act), tmp_dims)
    tmp_dims = [1] * (3) + dims[3 : len(dims)]
    prev_caps_lengths_tiled = tf.tile(prev_caps_lengths, tmp_dims)

    # Prepare routing weights
    tmp_dims = dims[0:4] + [1] * (len(dims) - 4)
    routing_weights_reshape = tf.reshape(curr_act[1], tmp_dims)
    tmp_dims = [1] * (4) + dims[4 : len(dims)]
    routing_weights_reshape_tiled = tf.tile(routing_weights_reshape, tmp_dims)

    # Prepare output of current caps layer
    tmp_dims = [1, 1, 1, dims[3]] + [1] * (len(dims) - 4)
    curr_caps_lengths = tf.reshape(compute_vectors_length(curr_act[0]), tmp_dims)
    tmp_dims = dims[0:3] + [1] + dims[4 : len(dims)]
    curr_caps_lengths_tiled = tf.tile(curr_caps_lengths, tmp_dims)

    # Calculate routing path visualization
    tmp = tf.multiply(routing_weights_reshape_tiled, curr_caps_lengths_tiled)
    all_paths = tf.multiply(prev_caps_lengths_tiled, tmp)
    all_paths_average = tf.reduce_sum(all_paths, axis=2)

    # Preprocess input image
    input_img_width, input_img_height = prep_img.shape[1:3]
    prep_img = prep_img[0]

    # Move preprocessed image to RGB domain (if not)
    if img_mode != "RGB":
        prep_img = np.repeat(prep_img[:, :] * 255.0, 3, axis=2)
    else:
        prep_img *= 255.0
    # Add alpha channel and convert to image
    prep_img = np.dstack((prep_img, np.full(prep_img.shape[:-1], 255.0))).astype("int8")
    prep_img = pil.fromarray(np.uint8(prep_img))

    # Prepare new image to contain capsules' activations
    new_img = pil.new("RGB", (input_img_width * dims[-1], input_img_height))

    # Save outputs as a single image
    all_paths_average = all_paths_average.numpy()
    all_paths_average = np.interp(
        all_paths_average,
        (all_paths_average.min(), all_paths_average.max()),
        (0.0, 1.0),
    )
    for i in range(all_paths_average.shape[-1]):
        rpv = all_paths_average[:, :, i]
        heatmap = pil.fromarray(np.uint8(matplotlib.cm.jet(rpv) * 255))

        # Rescale and blend
        heatmap = heatmap.resize((input_img_width, input_img_height))
        ccaps_img = pil.blend(prep_img, heatmap, alpha=0.7)

        new_img.paste(ccaps_img, (i * input_img_width, 0))

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"out.jpeg"))

    return {
        "Routing Path Visualization": {
            "filename": "out.jpeg",
            "rows": 1,
            "cols": dims[-1],
            "chunk_width": input_img_width,
            "chunk_height": input_img_height,
        }
    }


def mask_out_process(act, img_mode, layer, model_params, layer_img_dir):
    # Extract from batch and convert to numpy
    act = act[0]
    act_numpy = act.numpy()

    # Get next layer ( suppose decoder (HACK) )
    next_layer = layer._outbound_nodes[1].outbound_layer

    # Prepare dictionary for outputs
    out = {}

    # === MANIPULATIONS ON MAGNITUDE ===
    n_manip = 11

    # Prepare new output image
    new_img_width = model_params["input_shape"][0] * n_manip
    new_img_height = model_params["input_shape"][1]
    new_img = pil.new(img_mode, (new_img_width, new_img_height))

    # Edit tensor values and feed the activation forward to get reconstruction
    for i_r, r in enumerate(np.linspace(0, 1, n_manip)):
        r = round(r, 1)
        # Rescale magnitude value
        act_to_feed = np.copy(act_numpy)
        act_to_feed *= r / compute_vectors_length(act_to_feed)

        # Convert back to tensor and feed-forward
        act_to_feed = tf.expand_dims(tf.convert_to_tensor(act_to_feed), 0)
        reconstructed_image = next_layer(act_to_feed)

        # Pre-process reconstruction
        reconstructed_image = tf.squeeze(reconstructed_image)
        reconstructed_image = tf.multiply(reconstructed_image, 255.0)
        reconstructed_image = reconstructed_image.numpy().astype("int8")

        # Convert to image and paste to the new image
        reconstructed_image = pil.fromarray(reconstructed_image, mode=img_mode)
        new_img.paste(
            reconstructed_image, (model_params["input_shape"][0] * i_r, 0,),
        )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"magnitude.jpeg"))

    # Prepare partial output
    out["1) Manipulations on magnitude"] = {
        "filename": "magnitude.jpeg",
        "rows": 1,
        "cols": n_manip,
        "chunk_width": model_params["input_shape"][0],
        "chunk_height": model_params["input_shape"][1],
    }

    # === MANIPULATIONS ON DIMENSIONS ===
    # Prepare variables for the iterations
    start = np.argmax(act != 0)
    n_dims = act.shape[0] // model_params["n_class"]
    n_manip = 11

    # Prepare new output image
    new_img_width = model_params["input_shape"][0] * n_manip
    new_img_height = model_params["input_shape"][1] * n_dims
    new_img = pil.new(img_mode, (new_img_width, new_img_height))

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
            reconstructed_image = reconstructed_image.numpy().astype("int8")

            # Convert to image and paste to the new image
            reconstructed_image = pil.fromarray(reconstructed_image, mode=img_mode)
            new_img.paste(
                reconstructed_image,
                (
                    model_params["input_shape"][0] * i_r,
                    model_params["input_shape"][1] * (i_dim - start),
                ),
            )

    # Save the new image
    new_img.save(os.path.join(layer_img_dir, f"dimensions.jpeg"))

    out["2) Manipulations on dimensions"] = {
        "filename": "dimensions.jpeg",
        "rows": n_dims,
        "cols": n_manip,
        "chunk_width": model_params["input_shape"][0],
        "chunk_height": model_params["input_shape"][1],
    }

    return out


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
    plt.close("all")
    return {"outs": "out.png"}


def compute_step(model, model_params, prep_img, img_mode, req_out_dir):
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

    layer_pack = list(zip(processable_layers, activations))
    for i, ((layer, layer_type), act) in enumerate(layer_pack):
        # Get layer config
        layer_conf = layer.get_config()

        # Prepare directory for the layer
        layer_name = layer_conf["name"]
        layer_img_dir = os.path.join(req_out_dir, layer_name)
        os.mkdir(layer_img_dir)

        # === PROCESS LAYER ACTIVATION BASED ON TYPE ===
        if layer_type == "CONVOLUTIONAL":
            outs[layer_name] = conv_out_process(act, img_mode, layer_img_dir)
        elif layer_type == "PRIMARY_CAPS":
            outs[layer_name] = pcap_out_process(
                act, prep_img, img_mode, layer, layer_conf, layer_img_dir
            )
        elif layer_type == "CLASS_CAPS":
            curr_lp = layer_pack[i]
            prev_lp = layer_pack[i - 1]
            outs[layer_name] = ccap_out_process(
                prev_lp, curr_lp, prep_img, img_mode, layer_img_dir
            )
        elif layer_type == "MASK":
            outs[layer_name] = mask_out_process(
                act, img_mode, layer, model_params, layer_img_dir
            )
        else:
            raise TypeError(f"Layer type '{layer_type}' cannot be computed.")

    # Add layers outputs
    out_info["layers_outs"] = outs

    # Process model's output
    model_out_dir = os.path.join(req_out_dir, "model_out")
    os.mkdir(model_out_dir)

    model_outs = model(prep_img)
    if type(model_outs) == list:
        predictions = model_outs[0][0].numpy()
    else:
        predictions = model_outs[0].numpy()

    out_info["model_out"] = model_out_process(predictions, model_out_dir)

    return out_info
