# About
This folder contains Python files of CapsNet simple implementation in Keras and a simple model created as an example for the project. Feel free to play with the values of the model, the layers or just retrain it as it is.

## Usage
Note: the following instructions require Python 3 installed on your system.

In order to execute the training, you need to have installed 2 Python modules:
- Latest version of TensorFlow 2 at the moment of writing (2.2.0);
- Latest version of TensorFlowJS (used for weights and model save only) at the moment of writing (2.0.1.post1);

Those requirements are included in the `requirements.txt` file in the main folder, so be sure to have done:
```
pip3 install -r requirements.txt
```

Then execute the training process with:
```
python3 main.py
```