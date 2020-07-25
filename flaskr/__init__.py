"""
Flask Server's Factory
Author: Antonio Strippoli
"""
import os
import sys

from flask import Flask, render_template
from flask_json import FlaskJSON

# Create app
app = Flask(__name__)
json = FlaskJSON(app)

# Save some directories for later use
# (TODO: it could be usefull to have a way to provide the capsnet_trainer directory)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trainer_dir = os.path.join(project_dir, "capsnet_trainer")
data_dir = os.path.join(project_dir, "flaskr", "data")

# Add _share folder to sys.path
sys.path.append(os.path.join(trainer_dir, "_share"))

# Import API routes
from flaskr import api_routes

# Clean old cached images (if any)
api_routes.api_cleanImages()

# Simple route for the index
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
