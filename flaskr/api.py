"""
Core for the backend of the project.
Author: Antonio Strippoli
"""
# General imports and define current directory
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "data")
img_dir = os.path.join(data_dir, "images")

# Import TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model

# Custom layers and functions imports
sys.path.append(os.path.join(os.path.dirname(curr_dir), "capsnet_trainer"))
from capsnet import CapsuleNet
from capslayers import compute_vectors_length
from utils import pickle_load

# Temporary imports for number visualization
import numpy as np
import PIL.Image as pil


def buildNN(image):
    """
    Returns a JSON with the layers' name
    and the predicted value for the given image
    """
    # Create images directory (if not exists)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    # Save image for later visualization
    pil.fromarray(image).save(os.path.join(img_dir, "curr_image.jpeg"))

    # Preprocess image
    prep_image = image.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Load model params
    model_params = pickle_load(os.path.join(data_dir, "model_params.pkl"))
    model_params["batch_size"] = 1

    # Prepare model
    _, model = CapsuleNet(**model_params)

    for weights_filename in os.listdir(os.path.join(data_dir, "weights")):
        # Prepare images directory for the current instance
        instance_name = os.path.splitext(weights_filename)[0]
        instance_img_dir = os.path.join(img_dir, instance_name)
        if not os.path.exists(instance_img_dir):
            os.mkdir(instance_img_dir)

        # Load weights
        model.load_weights(os.path.join(data_dir, "weights", weights_filename))

        # Since we are not interested in every layer's output, we filter them
        def is_layer_processable(layer):
            keys = ["conv", "caps", "len", "mask"]
            layer_name = layer.name
            for key in keys:
                if key in layer_name:
                    return True
            return False

        # Create a temporary model to retrieve all the layers' output
        processable_layers = list(filter(is_layer_processable, model.layers))
        layers_outputs = [layer.output for layer in processable_layers]
        activation_model = Model(model.input, layers_outputs)

        # Get layers activations
        activations = activation_model(prep_image)

        for i, (layer, act) in enumerate(zip(processable_layers, activations)):
            # Get layer config
            config = layer.get_config()
            # Extract from batch
            act = act[0]

            # Prepare directory for the layer
            layer_name = config["name"]
            layer_img_dir = os.path.join(instance_img_dir, f"{i}_{layer_name}")
            if not os.path.exists(layer_img_dir):
                os.mkdir(layer_img_dir)

            # === PROCESS LAYER ACTIVATION BASED ON TYPE ===

            # Convolutional layer?
            if "conv" in layer_name:
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
            elif "primary" in layer_name and "caps" in layer_name:
                # =====
                # Vectors are high-dimensional,
                # the only interesting way to visualize is in which area the capsule
                # activated, so we visualize capsules' length
                # =====
                # Get new features' dimension
                feature_dim = int(
                    (layer.input_shape[1] - config["kernel_size"] + 1)
                    / config["strides"]
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
                new_img_width = act.shape[0] * act.shape[3] + margin * (
                    act.shape[3] + 1
                )
                new_img_height = act.shape[1] * act.shape[2] + margin * (
                    act.shape[2] + 1
                )
                new_img = pil.new("L", (new_img_width, new_img_height))

                # Save each feature as image
                for feature_i in range(act.shape[2]):
                    for dim_i in range(act.shape[3]):
                        curr_feature = act[:, :, feature_i, dim_i]
                        """
                        pil.fromarray(curr_feature).convert("L").save(
                            os.path.join(layer_img_dir, f"{feature_i}-{dim_i}.jpeg")
                        )
                        """
                        # TEST
                        curr_feature = pil.fromarray(curr_feature).convert("L")
                        new_img.paste(
                            curr_feature,
                            (
                                dim_i * act.shape[0] + margin * (dim_i + 1),
                                feature_i * act.shape[1] + margin * (feature_i + 1),
                            ),
                        )

                new_img.resize(
                    tuple(tmp * 10 for tmp in new_img.size), pil.NEAREST
                ).save(os.path.join(layer_img_dir, f"compact.jpeg"))

                """
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
            elif "caps" in layer_name:
                pass

            # Mask layer?
            elif "mask" in layer_name:
                # Get next layer (suppose decoder)
                next_layer = layer._outbound_nodes[1].outbound_layer

                # Edit tensor values and feed the activation forward to get reconstruction
                start = np.argmax(act != 0)
                n_dims = act.shape[0] // model_params["n_class"]
                act_numpy = act.numpy()

                for i_dim in range(start, start + n_dims):
                    # TODO: Could it be interesting to be able to manipulate also these values?
                    for r in np.linspace(-0.25, 0.25, 11):
                        r = round(r, 2)
                        # Edit dimension value
                        act_to_feed = np.copy(act_numpy)
                        act_to_feed[i_dim] += r

                        # Convert back to tensor and feed-forward
                        act_to_feed = tf.expand_dims(
                            tf.convert_to_tensor(act_to_feed), 0
                        )
                        reconstructed_image = next_layer(act_to_feed)

                        # Feed-forward and prepare reconstruction
                        reconstructed_image = tf.squeeze(reconstructed_image)
                        reconstructed_image = tf.multiply(reconstructed_image, 255.0)
                        reconstructed_image = reconstructed_image.numpy()

                        # Save as image
                        pil.fromarray(reconstructed_image).convert("L").save(
                            os.path.join(layer_img_dir, f"{i_dim-start}-({r}).jpeg")
                        )

    """
    # print(prediction)
    print(np.argmax(prediction, 1)[0])
    
    img_reconstructed = tf.multiply(img_reconstructed, 255.0)
    img_reconstructed = tf.squeeze(img_reconstructed)
    img_reconstructed = img_reconstructed.numpy()
    pil.fromarray(img_reconstructed).show()
    """
