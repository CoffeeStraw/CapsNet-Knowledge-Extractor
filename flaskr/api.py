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
        def processable_layer(layer):
            keys = ["conv", "caps"]
            layer_name = layer.name
            for key in keys:
                if key in layer_name:
                    return True
            return False

        # Create a temporary model to retrieve all the layers' output
        layers_outputs = [
            layer.output for layer in filter(processable_layer, model.layers)
        ]
        activation_model = Model(model.input, layers_outputs)

        # Get layers activations
        activations = activation_model(prep_image)

        for layer_name, act in zip([out.name for out in layers_outputs], activations):
            # Extract from batch
            act = act[0]

            # Prepare directory for the layer
            layer_name = layer_name.lower().split("/")[0]
            layer_img_dir = os.path.join(instance_img_dir, layer_name)
            if not os.path.exists(layer_img_dir):
                os.mkdir(layer_img_dir)

            # === Process layer activation based on type ===
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
                pass

            # Class Caps layer?
            elif "caps" in layer_name:
                pass

        """
        # print(prediction)
        print(np.argmax(prediction, 1)[0])
        
        img_reconstructed = tf.multiply(img_reconstructed, 255.0)
        img_reconstructed = tf.squeeze(img_reconstructed)
        img_reconstructed = img_reconstructed.numpy()
        pil.fromarray(img_reconstructed).show()
        """
        quit()
