"""
API routes for flask, not containing any computation on Neural Network.
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import shutil
import importlib
import PIL.Image as pil

# Flask imports
from flask import request
from flask_json import as_json, JsonError

# Project imports (Flask related)
from flaskr import app, trainer_dir, data_dir

# Neural Networks' functions
from .api_nn import load_model, get_processable_layers, compute_step

# Utilities
from utils import load_dataset


@app.route("/api/getModels", methods=["GET"])
@as_json
def api_getModels():
    """
    Returns every model's name in the trainer folder,
    along with its processable layers and training steps.

    This function is used by the front-end to gather
    informations about what can be visualized.
    """
    models = {}

    # Acquire models' name from trainer_dir, checking for every directory except '_share'
    models_name = [tmp for tmp in list(os.walk(trainer_dir))[0][1] if tmp != "_share"]

    # Add models' informations
    for name in models_name:
        # === ADD TRAINING STEPS (if found) ===
        try:
            training_steps = os.listdir(os.path.join(data_dir, name, "weights"))
        except FileNotFoundError:
            training_steps = []
        models[name] = {"training_steps": training_steps}

        # === ADD PROCESSABLE LAYERS ===
        model, _ = load_model(name)

        # Get processable layers
        processable_layers = get_processable_layers(model.layers)
        models[name]["layers"] = processable_layers

    return models


@app.route("/api/computeStep", methods=["POST"])
@as_json
def api_computeStep():
    """
    Computes all processable outputs for a model at a given training step.
    It is required a JSON with a structure like the following:
    {
        "model": "Simple",
        "step": "trained"
    }

    Returns a JSON `{'status': 200}` if everything went fine,
    a JSON containing an error if something went wrong.
    """
    # Unpack data from request
    data = request.json
    model_name = data["model"]
    training_step = data["step"] + ".h5"

    # Prepare model
    model, model_params = load_model(model_name)
    model.load_weights(os.path.join(data_dir, model_name, "weights", training_step))

    # Check if output directory exists
    step_img_dir = os.path.join(data_dir, model_name, "images")
    if not os.path.exists(step_img_dir):
        os.mkdir(step_img_dir)

    # Save all the outputs for the given training step
    compute_step(model, model_params, step_img_dir)

    return {"status": 200}


@app.route("/api/cleanImages")
def api_cleanImages():
    """
    Clean all the images pre-calculated for every model.
    This has to be done every time a new image is submitted to the network.
    """
    # Delete every images folder
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name, "images")
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    # Delete curr_image picture
    curr_image_path = os.path.join(data_dir, "curr_image.jpeg")
    if os.path.exists(curr_image_path):
        os.remove(curr_image_path)


@app.route("/api/setImage", methods=["POST"])
@as_json
def api_setImage():
    """
    Set the current image to be processed by the Neural Network.
    You can either pass an image in a JSON like:
    {
        "image": BinaryImage
    }
    Or you can pass a supported dataset name and an arbitrary index like:
    {
        "dataset": "MNIST",
        "index": 0
    }

    Returns a JSON `{'status': 200}` if everything went fine,
    a JSON containing an error if something went wrong.
    """
    data = request.json

    # Image is passed?
    if "image" in data:
        pass
    elif "dataset" in data and "index" in data:
        # TODO: Exception for invalid dataset should be checked here
        _, (x_test, _) = load_dataset(name=data["dataset"])
        # TODO: Exception for invalid index should be checked here
        image = x_test[data["index"]][:, :, 0] * 255.0
        pil.fromarray(image).convert("L").save(
            os.path.join(data_dir, "curr_image.jpeg")
        )
    else:
        raise JsonError(error_description="Invalid request, check the documentation.")

    return {"status": 200}
