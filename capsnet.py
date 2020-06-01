import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras import Model

from capslayers import PrimaryCaps, ClassCaps, compute_vectors_length


class CapsuleNet(Model):
    """Capsule Network model implementation, used for MNIST dataset training.

    The structure and the values used for the implementation
    are taken from the [official paper](https://arxiv.org/abs/1710.09829).

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

        # Layer 1: ReLU Convolutional Layer
        self.conv1 = Conv2D(input_shape=input_shape, data_format='channels_last',
                            filters=256, kernel_size=9, strides=1,
                            padding='valid', activation='relu', name='conv1')

        # Layer 2: PrimaryCaps Layer
        self.primary_caps = PrimaryCaps(
            n_capsules=32, out_dim_capsule=8, kernel_size=9, strides=2, padding='valid', name='primary_caps')

        # Layer 3: DigitCaps Layer: since routing it is computed only
        # between two consecutive capsule layers, it only happens here
        self.digit_caps = ClassCaps(
            n_capsules=n_class, out_dim_capsule=16, r_iter=r_iter, name='digit_caps')

        # Layer 3.5: A convenience layer to calculate vectors' length
        self.vec_len = Lambda(compute_vectors_length, name='vec_len')

    def call(self, inputs, training=False):
        """CapsNet's forward pass.
        """
        # Conv1 shape out:          (batch_size, 20, 20, 256)
        x = self.conv1(inputs)

        # PrimaryCaps shape out:    (batch_size, 1152, 8)
        x = self.primary_caps(x)

        # DigitCaps shape out:      (batch_size, 10, 16)
        x = self.digit_caps(x)

        # VecLen shape out:         (batch_size, 10)
        x = self.vec_len(x)

        return x

    def summary(self, batch_size):
        """Override summary to infer output shape of each layer.
        """
        tmp = Input(shape=self.in_shape, batch_size=batch_size, name="input_images")
        return Model(inputs=tmp, outputs=self.call(tmp), name=self.name).summary()
