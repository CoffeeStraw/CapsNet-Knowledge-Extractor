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
    def say_hello():
        """
        API to get architecture of the trained NN, as well as
        weights, biases and outputs of each layer.

        Note that an image has to be passed in the POST request.
        """
        # TODO: Get Image file
        pass

        # TODO: Preprocess Image file
        pass

        return buildNN([])

    # TESTING, to be removed
    buildNN([])
    quit()

    return app
