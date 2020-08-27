"""
Flask Server's starter
Author: Antonio Strippoli
"""
import os
import sys
import shutil
from multiprocessing import Value

from flask import Flask, render_template
from flask_json import FlaskJSON

# Create app
app = Flask(__name__)
json = FlaskJSON(app)
req_counter = Value("i", 0)

# Save some paths for later use
# TODO: it could be usefull to have a way to provide the capsnet_trainer directory
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = {
    "trainer": os.path.join(project_path, "capsnet_trainer"),
    "out": os.path.join(project_path, "flaskr", "static", "img", "outs"),
}

# Check for directories
if not os.path.exists(paths["trainer"]):
    raise NotADirectoryError(
        f"The trainer directory was not found ({paths['trainer']})."
    )

# Clean up old outputs (if any)
if os.path.exists(paths["out"]):
    shutil.rmtree(paths["out"])
os.mkdir(paths["out"])

# Add _share folder to sys.path
sys.path.append(os.path.join(paths["trainer"], "_share"))

# Import API routes
from flaskr import api_routes

# Simple route for the index
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
