import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Reshape
from tensorflow.keras import Model

from capslayers import PrimaryCaps, ClassCaps, compute_vectors_length, mask


class CapsuleNet(Model):
    """Capsule Network model implementation, used for MNIST dataset training.

    The structure used for the implementation
    are taken from the [official paper](https://arxiv.org/abs/1710.09829).

    The values are changed in order to simplify the model.

    Arguments:
        input_shape: 3-Dimensional data shape (width, height, channels).
        n_class: Number of classes.
        r_iter: Number of routing iterations.
    """

    def __init__(self, input_shape, n_class, r_iter=3, name="CapsNet_MNIST"):
        """Define the layers, using Keras model subclassing API.
        """
        super(CapsuleNet, self).__init__(name=name)
        self.in_shape = input_shape

        # --- Encoder ---
        # Layer 1: ReLU Convolutional Layer
        self.conv1 = Conv2D(input_shape=input_shape, data_format='channels_last',
                            filters=128, kernel_size=9, strides=1,
                            padding='valid', activation='relu', name='conv1')

        # Layer 2: PrimaryCaps Layer
        self.primary_caps = PrimaryCaps(
            n_capsules=16, out_dim_capsule=8, kernel_size=9, strides=2, padding='valid', name='primary_caps')

        # Layer 3: DigitCaps Layer: since routing it is computed only
        # between two consecutive capsule layers, it only happens here
        self.digit_caps = ClassCaps(
            n_capsules=n_class, out_dim_capsule=16, r_iter=r_iter, name='digit_caps')

        # Layer 4: A convenience layer to calculate vectors' length
        self.vec_len = Lambda(compute_vectors_length, name='vec_len')

        # --- Decoder ---
        # Layer 1: A convenience layer to compute the masked capsules' output
        self.mask = Lambda(mask, name="mask")

        # Layer 2-4: Three Dense layer for the image reconstruction
        self.dense1 = Dense(
            256,
            activation='relu',
            input_dim=16*n_class,
            name="dense_1")

        self.dense2 = Dense(
            512,
            activation='relu',
            name="dense_2")

        self.dense3 = Dense(
            tf.math.reduce_prod(input_shape),
            activation='sigmoid',
            name="dense_3")

        # Layer 5: Reshape the output as the image provided in input
        self.reshape = Reshape(target_shape=input_shape,
                               name='img_reconstructed')

    def call(self, inputs, training=False):
        """CapsNet's forward pass.
        """
        print(training)
        print(inputs, type(inputs) is tuple, end='\n\n')
        training = type(inputs) is tuple
        # During training, input for the decoder is masked by labels y
        if training:
            inputs, y = inputs

        # --- Encoder ---
        # Conv1 shape out:          (batch_size, 20, 20, 256)
        conv_out = self.conv1(inputs)

        # PrimaryCaps shape out:    (batch_size, 1152, 8)
        caps1_out = self.primary_caps(conv_out)

        # DigitCaps shape out:      (batch_size, 10, 16)
        caps2_out = self.digit_caps(caps1_out)

        # VecLen shape out:         (batch_size, 10)
        out1 = self.vec_len(caps2_out)

        # --- Decoder ---
        caps2_masked = self.mask((caps2_out, y) if training else caps2_out)

        out2 = self.dense1(caps2_masked)
        out2 = self.dense2(out2)
        out2 = self.dense3(out2)

        out2 = self.reshape(out2)

        return (out1, out2)

    def summary(self, batch_size):
        """Override summary to infer output shape of each layer.
        """
        tmp = Input(shape=self.in_shape,
                    batch_size=batch_size, name="input_images")
        return Model(inputs=tmp, outputs=self.call(tmp), name=self.name).summary()
