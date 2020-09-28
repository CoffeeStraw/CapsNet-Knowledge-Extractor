# About
This folder contains the codebase of our CapsNet simple implementation in Keras, along with the models studied in the thesis.

All the shared files of the models are in the [`./_share`](/share) directory.

You can edit one of the existing model or create another one by creating a new directory for it. Remember that the name of the directory should represent the model's name and will be used in the visualization (however, you can change that in your main.py file).

## Usage
The requirements for the entire project are included in the [`../requirements.txt`](requirements.txt) file in the main folder, so be sure to have done:
```
pip3 install -r requirements.txt
```

Then execute the training process of one of the models with:
```
python3 main.py
```

All the outputs are by default produced in a dedicated `/output` folder, inside your model's directory.