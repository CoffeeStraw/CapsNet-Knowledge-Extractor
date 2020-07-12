"""
Core for the backend of the project.
Author: Antonio Strippoli
"""
# General imports and define current directory
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Import TensorFlow & Keras
import tensorflow as tf

# Custom layers and functions imports
sys.path.append(
    os.path.join(os.path.dirname(curr_dir), "capsnet_trainer")
)  # Enable imports from capsnet_trainer
from capsnet import CapsuleNet
from utils import pickle_load


def buildNN(image):
    """
    Returns a JSON with preprocess Neural Network
    with the given image.
    """
    data_dir = os.path.join(curr_dir, "data")

    # Load model params
    model_params = pickle_load(os.path.join(data_dir, "model_params.pkl"))

    # TODO: Load every training istance (just one now for testing)
    filename = "1-0.h5"
    # TODO: Later we will use eval_model, not training_model
    model, _ = CapsuleNet(**model_params)
    model.load_weights(os.path.join(data_dir, "training", filename))

    # Debug: print loaded model
    print(model)
