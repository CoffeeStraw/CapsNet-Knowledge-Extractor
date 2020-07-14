"""
Core for the backend of the project.
Author: Antonio Strippoli
"""
# General imports and define current directory
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "data")

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
    Returns a JSON with preprocess Neural Network
    with the given image.
    """
    # Preprocess image
    prep_image = image.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    # Load model params
    model_params = pickle_load(os.path.join(data_dir, "model_params.pkl"))
    model_params["batch_size"] = 1

    # Prepare model
    _, model = CapsuleNet(**model_params)
    filename = "2-0.h5"
    model.load_weights(os.path.join(data_dir, "training", filename))

    # Create a temporary model to retrieve all the layers' output
    # Access all the layers outputs, except first one (input) and last one (dense layer, useless to visualize)
    layer_outputs = [layer.output for layer in model.layers[1:-1]]
    activation_model = Model(model.input, layer_outputs)

    # DEBUG: show first image of mnist dataset
    pil.fromarray(image).show()

    # Save convolutional network images
    activations = activation_model(prep_image)
    conv_act = tf.multiply(activations[0][0], 255.0)

    for i in range(conv_act.shape[-1]):
        filter = conv_act[:, :, i].numpy()
        pil.fromarray(filter).show()
        quit()

    """
    # print(prediction)
    print(np.argmax(prediction, 1)[0])
    
    img_reconstructed = tf.multiply(img_reconstructed, 255.0)
    img_reconstructed = tf.squeeze(img_reconstructed)
    img_reconstructed = img_reconstructed.numpy()
    pil.fromarray(img_reconstructed).show()
    """
