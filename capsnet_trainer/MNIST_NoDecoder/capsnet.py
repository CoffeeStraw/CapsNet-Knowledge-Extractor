import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Reshape
from tensorflow.keras.models import Sequential, Model

from capslayers import PrimaryCaps, DenseCaps, compute_vectors_length, mask


def CapsuleNet(input_shape, n_class, name="CapsuleNetwork"):
    """Capsule Network model implementation, used for MNIST dataset training.

    The model has been adapted from
    the [official paper](https://arxiv.org/abs/1710.09829).

    Arguments:
        input_shape: 3-Dimensional data shape (width, height, channels).
        n_class: Number of classes.
    """
    # --- Encoder ---
    x = Input(shape=input_shape, dtype=tf.float32)

    # Layer 1: ReLU Convolutional Layer
    conv1 = Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding="valid",
        activation="relu",
        name="conv1",
    )(x)

    # Layer 2: PrimaryCaps Layer
    primary_caps = PrimaryCaps(
        n_caps=32,
        dims_caps=8,
        kernel_size=9,
        strides=2,
        padding="valid",
        activation="relu",
        name="primary_caps",
    )(conv1)

    # Layer 3: DigitCaps Layer: since routing it is computed only
    # between two consecutive capsule layers, it only happens here
    digit_caps = DenseCaps(n_caps=n_class, dims_caps=16, name="digit_caps")(
        primary_caps
    )[0]

    # Layer 4: A convenience layer to calculate vectors' length
    vec_len = Lambda(compute_vectors_length, name="vec_len")(digit_caps)

    # Models for training and evaluation (prediction)
    train_model = Model(inputs=[x], outputs=[vec_len], name=f"{name}_training")
    eval_model = Model(inputs=x, outputs=[vec_len], name=name)

    return train_model, eval_model
