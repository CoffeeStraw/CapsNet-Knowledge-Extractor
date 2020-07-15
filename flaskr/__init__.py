"""
Flask Server's Factory
Author: Antonio Strippoli
"""
import os
from flask import Flask, render_template

# API import
from .api import buildNN


def create_app():
    # Create app
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/buildNN", methods=["POST"])
    def api_buildNN():
        """
        API to get architecture of the trained NN, as well as
        weights, biases and outputs of each layer.

        Note that an image has to be passed in the POST request.
        """
        # TODO: Get Image file from POST req
        
        # Testing: get image from mnist dataset
        from tensorflow.keras.datasets import mnist
        _, (x_test, _) = mnist.load_data()
        index_img = 0  # Arbitrary index

        return buildNN(x_test[index_img])

    # TESTING, to be removed
    # api_buildNN()
    # quit()

    return app
