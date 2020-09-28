<h1 style="text-align: center;">Capsule Network<br/>Knowledge Extractor</h1>

<p style="text-align: center;">My Bachelor's Thesis. <i>Capsule Network Knowledge Extractor</i> is a software to easily produce <b>network visualizations</b> of a Capsule Network created with <b>Keras under TensorFlow 2.0</b>.</p>

<img src="./thesis/img/cnke_overview.png">

## Running Locally
The project requires **Python 3** installed on your system. Install the dependencies:
```
pip3 install -r requirements.txt
```

Run a local server:
```
python3 run.py
```

Finally, navigate to [127.0.0.1:5000](https://127.0.0.1:5000) to use the software.

Additionally, you can edit the Capsule Network model or re-train the current one. For more informations about that, please check [`./capsnet_trainer/`](capsnet_trainer).

## Visualization Techniques

<div style="text-align: center;">
    <img src="./thesis/img/visualization_conv.png" width="50%">
    <p><i>Convolutional layer's feature maps.</i></p>
</div>

<hr>

<div style="text-align: center;">
    <img src="./thesis/img/visualization_pcaps.png" width="50%">
    <p><i>Primary Capsules' entities localization.</i></p>
</div>

<hr>

<div style="text-align: center;">
    <img src="./thesis/img/visualization_dcaps_square.png" width="90%">
    <p><i>Improved Routing Path Visualization Technique.</i></p>
</div>

<hr>

<div style="text-align: center;">
    <img src="./thesis/img/visualization_decoder_1.png" width="60%">
    <img src="./thesis/img/visualization_decoder_2.png" width="60%">
    <p><i>Dense Capsules' magnitude and dimensions manipulated and visualized through the Decoder.</i></p>
</div>

<hr>

<div style="text-align: center;">
    <img src="./thesis/img/visualization_gradcam.png" width="35%">
    <p><i>Area of interests produced by Grad-CAM++.</i></p>
</div>

### Thesis
You can *download or view* the .pdf file of the thesis [here](./thesis/CNKE_Thesis.pdf). Please note that this file will never be updated, nor the latex files will be shared.

### License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.