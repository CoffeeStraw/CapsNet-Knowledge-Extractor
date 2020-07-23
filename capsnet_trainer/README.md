# About
This folder contains Python files of CapsNet simple implementation in Keras and two models created as an example for the project:
1. `Original` is the model which was introduced in the [official paper](https://arxiv.org/abs/1710.09829);
2. `Simple` is a lighter model, with less parameters. It has been created to have a simpler model with less output for the layers.

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

All the outputs are by default produced in `/flaskr/data`. To change that, check more info about the arguments you can pass to the script with:
```
python3 main.py -h
```