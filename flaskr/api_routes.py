"""
API routes for flask, not containing any computation on Neural Network.
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import importlib
import numpy as np
import PIL.Image as pil
from natsort import natsorted

# Flask imports
from flask import request
from flask_json import as_json, JsonError

# Project imports (Flask related)
from flaskr import app, req_counter, paths

# Neural Networks' functions
from .api_nn import load_model, get_processable_layers, compute_step

# Utilities
from .utils import img_transform


@app.route("/api/getModels", methods=["GET"])
@as_json
def api_getModels():
    """
    Returns every model's name in the trainer folder,
    along with its training steps and processable layers.

    This function is used by the front-end to gather
    informations about what can be visualized.
    """
    models = {}

    # Acquire models' name from trainer's path, checking for every directory except '_share'
    models_name = [
        tmp for tmp in list(os.walk(paths["trainer"]))[0][1] if tmp != "_share"
    ]

    # Add models' informations
    for name in models_name:
        # Add training steps (if found)
        try:
            training_steps = natsorted(
                map(
                    lambda step_name: step_name.replace(".h5", ""),
                    os.listdir(os.path.join(paths["trainer"], name, "outs", "weights")),
                )
            )
        except FileNotFoundError:
            training_steps = []

        models[name] = {"training_steps": training_steps}

        # Add processable layers
        model, _ = load_model(name)
        if model != None:
            models[name]["layers"] = get_processable_layers(model.layers)
        else:
            models[name]["layers"] = []

    return {"models": models}


@app.route("/api/computeStep", methods=["POST"])
@as_json
def api_computeStep():
    """
    Send an image to the wanted model at a given training step, saving all the outputs.
    The request's body is expected to have the following key/value pairs:
    {
        "model": "YourModelName",
        "step": "YourTrainingStep",
        "rotation": '0 or 90 or 180 or 270',
        "invert_colors": 'true or false'
    }

    Then, you must also provide one of the following, depending on your objective:
    
    1) Send a custom image:
        A file named "image" containing the data for the image.
    2) An additional key/value pair that specifies an arbitrary index
       for the test set used for the given model, like: "testset_index": 0

    Returns a JSON `{'status': 200}` if everything went fine,
    a JSON containing an error if something went wrong.
    """
    # Unpack data from request
    data = request.form
    model_name = data["model"]
    training_step = data["step"] + ".h5"
    rotation = int(data["rotation"])
    invert_colors = True if data["invert_colors"] == "true" else False

    # Prepare model
    model, model_params = load_model(model_name)

    # Preprocess image depending on request
    if "testset_index" in data:
        index = int(data["testset_index"])

        if model_params["dataset"] == "MNIST":
            from tensorflow.keras.datasets import mnist

            img = mnist.load_data()[1][0][index]
        elif model_params["dataset"] == "Fashion_MNIST":
            from tensorflow.keras.datasets import fashion_mnist

            img = fashion_mnist.load_data()[1][0][index]
        elif model_params["dataset"] == "CIFAR10":
            from tensorflow.keras.datasets import cifar10

            img = cifar10.load_data()[1][0][index]
        else:
            raise JsonError(error_description="Dataset not supported.")

        # Preprocess the image
        img = img_transform(img, rotation, invert_colors)
        img_mode = "RGB" if len(img.shape) == 3 else "L"

        prep_img = img
        if len(img.shape) == 2:
            prep_img = np.expand_dims(prep_img, -1)

        prep_img = np.expand_dims(prep_img, 0)
        prep_img = prep_img.astype("float32") / 255.0
    elif request.files:
        raise JsonError(error_description="Not implemented.")
    else:
        raise JsonError(
            error_description="Invalid JSON for '/computeStep' API, check the documentation."
        )

    # Get request counter and set output directory
    with req_counter.get_lock():
        req_number = req_counter.value
        req_counter.value += 1

    req_out_dir = os.path.join(paths["out"], f"{req_number}")
    os.mkdir(req_out_dir)

    # Save image for future visualization
    pil.fromarray(img, mode=img_mode).save(os.path.join(req_out_dir, "img.jpeg"))

    # HACK, pass an image to build the model and load the weights
    model(prep_img)
    model.load_weights(
        os.path.join(paths["trainer"], model_name, "outs", "weights", training_step)
    )

    return compute_step(model, model_params, prep_img, img_mode, req_out_dir)
