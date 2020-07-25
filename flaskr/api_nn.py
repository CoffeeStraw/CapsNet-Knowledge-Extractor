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

# Project's imports
from flaskr import paths

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model

# Utilities
from utils import pickle_load


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


def compute_step(model, model_params, prep_img, req_out_dir):
    """
    Computes and saves the outputs of the processable layers in the given model.
    """
    # Create a temporary model to retrieve all the layers' activations
    processable_layers = get_processable_layers(model.layers)
    processable_layers = [(model.get_layer(pl[0]), pl[1]) for pl in processable_layers]

    activation_model = Model(model.input, [pl[0].output for pl in processable_layers])
    activations = activation_model(prep_img)

    for (layer, layer_type), act in zip(processable_layers, activations):
        # Get layer config
        config = layer.get_config()
        # Extract from batch
        act = act[0]

        # Prepare directory for the layer
        layer_name = config["name"]
        layer_img_dir = os.path.join(req_out_dir, layer_name)
        if not os.path.exists(layer_img_dir):
            os.mkdir(layer_img_dir)

        # === PROCESS LAYER ACTIVATION BASED ON TYPE ===

        # Convolutional layer?
        if layer_type == "CONVOLUTIONAL":
            # Save features
            act = tf.multiply(act, 255.0)
            for i in range(act.shape[-1]):
                feature = act[:, :, i].numpy()
                pil.fromarray(feature).convert("L").save(
                    os.path.join(layer_img_dir, f"{i}.jpeg")
                )

            # Save filters (TODO?)
            pass

        # Primary Caps layer?
        elif layer_type == "PRIMARY_CAPS":
            # Get new features' dimension
            feature_dim = int(
                (layer.input_shape[1] - config["kernel_size"] + 1) / config["strides"]
            )
            # Reshape to get capsules' features
            act = tf.reshape(
                act,
                (
                    feature_dim,
                    feature_dim,
                    config["n_capsules"],
                    config["out_dim_capsule"],
                ),
            )
            # Rescale activations linearly
            min_value, max_value = np.min(act), np.max(act)
            act = (255 / (max_value - min_value)) * (act.numpy() - min_value)

            # Prepare new image to represent capsules
            margin = 2
            new_img_width = act.shape[0] * act.shape[3] + margin * (act.shape[3] + 1)
            new_img_height = act.shape[1] * act.shape[2] + margin * (act.shape[2] + 1)
            new_img = pil.new("L", (new_img_width, new_img_height))

            # Save each feature as image
            for i_feature in range(act.shape[2]):
                for i_dim in range(act.shape[3]):
                    curr_feature = act[:, :, i_feature, i_dim]
                    curr_feature = pil.fromarray(curr_feature).convert("L")

                    curr_feature.save(
                        os.path.join(layer_img_dir, f"{i_feature}-{i_dim}.jpeg")
                    )

                    """
                    # Save as a single image
                    new_img.paste(
                        curr_feature,
                        (
                            act.shape[0] * i_dim + margin * (i_dim + 1),
                            act.shape[1] * i_feature + margin * (i_feature + 1),
                        ),
                    )
                    """

            """
            # Save as a single image
            new_img.resize(tuple(tmp * 10 for tmp in new_img.size), pil.NEAREST).save(
                os.path.join(layer_img_dir, f"compact.jpeg")
            )
            """

            """ OLD CODE TO SAVE VECTORS' LENGTH, TO BE REMOVED?
            # Compute vectors' length and reshape
            act = tf.reshape(
                compute_vectors_length(act), (feature_dim, feature_dim, -1)
            )
            # Save as images with the same size of the inputs
            act = tf.multiply(act, 255.0)
            for i in range(act.shape[-1]):
                capsules_length = act[:, :, i].numpy()
                pil.fromarray(capsules_length).convert("L").resize(
                    image.shape, pil.NEAREST
                ).save(os.path.join(layer_img_dir, f"{i}.jpeg"))
            """

        # Class Caps layer?
        elif layer_type == "CLASS_CAPS":
            pass

        # Mask layer?
        elif layer_type == "MASK":
            # Get next layer ( suppose decoder (HACK) )
            next_layer = layer._outbound_nodes[1].outbound_layer

            # Prepare variables for the iterations
            act_numpy = act.numpy()
            start = np.argmax(act != 0)
            n_dims = act.shape[0] // model_params["n_class"]
            n_manip = 11

            # TEST, put all the reconstructions in one single image
            """
            margin = 2
            new_img_width = prep_img.shape[1] * n_manip + margin * (n_manip + 1)
            new_img_height = prep_img.shape[2] * n_dims + margin * (n_dims + 1)
            new_img = pil.new("L", (new_img_width, new_img_height))
            """

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

                    # Convert to image
                    reconstructed_image = pil.fromarray(reconstructed_image)
                    reconstructed_image = reconstructed_image.convert("L")

                    # Paste the reconstruction to the single image that we are building
                    """
                    new_img.paste(
                        reconstructed_image,
                        (
                            prep_img.shape[1] * i_r + margin * (i_r + 1),
                            prep_img.shape[2] * (i_dim - start)
                            + margin * (i_dim - start + 1),
                        ),
                    )
                    """

                    # Save as image
                    reconstructed_image.save(
                        os.path.join(layer_img_dir, f"{i_dim-start:02}-({i_r:02}).jpeg")
                    )

            # Save as a single img
            # new_img.save(os.path.join(layer_img_dir, f"compact.jpeg"))

    """
    # print(prediction)
    print(np.argmax(prediction, 1)[0])
    
    img_reconstructed = tf.multiply(img_reconstructed, 255.0)
    img_reconstructed = tf.squeeze(img_reconstructed)
    img_reconstructed = img_reconstructed.numpy()
    pil.fromarray(img_reconstructed).show()
    """
