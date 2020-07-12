import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Reshape
from tensorflow.keras.models import Sequential, Model

from capslayers import PrimaryCaps, ClassCaps, compute_vectors_length, mask


def CapsuleNet(input_shape, batch_size, n_class, name="CapsNet_MNIST"):
    """Capsule Network model implementation, used for MNIST dataset training.

    The structure used for the implementation
    are taken from the [official paper](https://arxiv.org/abs/1710.09829).

    The values are changed in order to simplify the model.

    Arguments:
        input_shape: 3-Dimensional data shape (width, height, channels).
        batch_size: Size of the batches of the inputs.
        n_class: Number of classes.
    """
    # --- Encoder ---
    x = Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: ReLU Convolutional Layer
    conv1 = Conv2D(
        input_shape=input_shape,
        data_format="channels_last",
        filters=128,
        kernel_size=9,
        strides=1,
        padding="valid",
        activation="relu",
        name="conv1",
    )(x)

    # Layer 2: PrimaryCaps Layer
    primary_caps = PrimaryCaps(
        n_capsules=16,
        out_dim_capsule=8,
        kernel_size=9,
        strides=2,
        padding="valid",
        name="primary_caps",
    )(conv1)

    # Layer 3: DigitCaps Layer: since routing it is computed only
    # between two consecutive capsule layers, it only happens here
    digit_caps = ClassCaps(n_capsules=n_class, out_dim_capsule=16, name="digit_caps")(
        primary_caps
    )

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
    decoder.add(Dense(256, activation="relu", input_dim=16 * n_class, name="dense_1"))

    decoder.add(Dense(512, activation="relu", name="dense_2"))

    decoder.add(
        Dense(tf.math.reduce_prod(input_shape), activation="sigmoid", name="dense_3")
    )

    # Layer 5: Reshape the output as the image provided in input
    decoder.add(Reshape(target_shape=input_shape, name="img_reconstructed"))

    # Models for training and evaluation (prediction)
    train_model = Model([x, y], [vec_len, decoder(masked_by_y)])
    eval_model = Model(x, [vec_len, decoder(masked)])

    return train_model, eval_model
