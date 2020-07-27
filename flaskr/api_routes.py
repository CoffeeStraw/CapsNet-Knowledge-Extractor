"""
API routes for flask, not containing any computation on Neural Network.
Author: Antonio Strippoli
"""
# General imports
import os
import sys
import importlib
import PIL.Image as pil
from natsort import natsorted

# Flask imports
from flask import request
from flask_json import as_json, JsonError

# Project imports (Flask related)
from flaskr import app, req_counter, paths

# Neural Networks' functions
from .api_nn import load_model, get_processable_layers, compute_step


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
                    lambda name: name.replace(".h5", ""),
                    os.listdir(os.path.join(paths["data"], name, "weights")),
                )
            )
        except FileNotFoundError:
            training_steps = []

        models[name] = {"training_steps": training_steps}

        # Add processable layers
        model, _ = load_model(name)
        models[name]["layers"] = get_processable_layers(model.layers)

    return {"models": models}


@app.route("/api/computeStep", methods=["POST"])
@as_json
def api_computeStep():
    """
    Send an image to the wanted model at a given training step, saving all the outputs.
    There are 2 different JSON that can be sent to this API:
    
    1) Send an image file in the request:
        {
            "model": "Simple",
            "step": "trained",
            "image": BinaryImage
        }
    2) Specify a supported dataset name and an arbitrary index for that:
        {
            "model": "Simple",
            "step": "trained",
            "dataset": "MNIST",
            "index": 0
        }

    Returns a JSON `{'status': 200}` if everything went fine,
    a JSON containing an error if something went wrong.
    """
    # Unpack data from request
    data = request.json
    model_name = data["model"]
    training_step = data["step"] + ".h5"

    # Preprocess image depending on request
    if "image" in data:
        raise JsonError(error_description="Not implemented.")
    elif "dataset" in data and "index" in data:
        if data["dataset"] == "MNIST":
            from tensorflow.keras.datasets import mnist

            img = mnist.load_data()[1][0][data["index"]]
            prep_img = img.reshape(1, 28, 28, 1).astype("float32") / 255.0
        else:
            raise JsonError(error_description="Dataset not supported.")
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
    pil.fromarray(img).convert("L").save(os.path.join(req_out_dir, "img.jpeg"))

    # Prepare model
    model, model_params = load_model(model_name)
    model.load_weights(
        os.path.join(paths["data"], model_name, "weights", training_step)
    )

    # Save all the outputs for the given training step
    compute_step(model, model_params, prep_img, req_out_dir)

    # Get all outputs' names
    layers_outs = {}
    for layer_name in [tmp for tmp in list(os.walk(req_out_dir))[0][1]]:
        imgs = []
        for img_name in natsorted(os.listdir(os.path.join(req_out_dir, layer_name))):
            imgs.append(img_name)
        layers_outs[layer_name] = imgs

    # Prepare output directory path for return
    req_out_dir = req_out_dir.replace("\\", "/")
    req_out_dir = req_out_dir[req_out_dir.find("/static") :] + "/"

    return {"out_dir": req_out_dir, "layers_outs": layers_outs, "status": 200}
