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

    # --- Decoder ---
    y = Input(shape=(n_class,))

    # Layer 1: A convenience layer to compute the masked capsules' output
    masked = Lambda(mask, name="masked")(
        digit_caps
    )  # Mask using the capsule with maximal length. For prediction
    masked_by_y = Lambda(mask, name="masked_by_y")(
        [digit_caps, y]
    )  # The true label is used to mask the output of capsule layer. For training

    # Layer 2-4: Three Dense layer for the image reconstruction
    decoder = Sequential(name="decoder")
    decoder.add(Dense(512, activation="relu", input_dim=16 * n_class, name="dense_1"))
    decoder.add(Dense(1024, activation="relu", name="dense_2"))
    decoder.add(
        Dense(tf.math.reduce_prod(input_shape), activation="sigmoid", name="dense_3")
    )

    # Layer 5: Reshape the output as the image provided in input
    decoder.add(Reshape(target_shape=input_shape, name="img_reconstructed"))

    # Models for training and evaluation (prediction)
    train_model = Model(
        inputs=[x, y], outputs=[vec_len, decoder(masked_by_y)], name=f"{name}_training"
    )
    eval_model = Model(inputs=x, outputs=[vec_len, decoder(masked)], name=name)

    return train_model, eval_model
