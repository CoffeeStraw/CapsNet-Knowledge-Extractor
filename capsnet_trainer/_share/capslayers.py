import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, Reshape, Lambda
from tensorflow.keras.backend import (
    epsilon,
    one_hot,
    argmax,
    batch_flatten,
    expand_dims,
)
from tensorflow.keras import initializers


class PrimaryCaps(Layer):
    """
    A PrimaryCaps layer. More info To Be Added.
    """

    def __init__(
        self,
        n_caps,
        dims_caps,
        kernel_size,
        strides,
        padding,
        activation=None,
        **kwargs,
    ):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_caps": self.n_caps,
                "dims_caps": self.dims_caps,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
            }
        )
        return config

    def build(self, input_shape):
        assert (
            len(input_shape) >= 4
        ), "The input Tensor of a PrimaryCaps should have shape=[None, width, height, channels]"

        # Apply Convolution n_caps times
        self.conv2d = Conv2D(
            filters=self.n_caps * self.dims_caps,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            name="primarycaps_conv2d",
        )

        # Reshape the convolutional layer output
        feature_dims = int((input_shape[1] - self.kernel_size + 1) / self.strides)
        self.reshape = Reshape(
            (feature_dims ** 2 * self.n_caps, self.dims_caps),
            name="primarycaps_reshape",
        )

        # Squash the vectors output
        self.squash = Lambda(squash, name="primarycaps_squash")

        self.built = True

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.reshape(x)
        return self.squash(x)


class ClassCaps(Layer):
    """A DigitCaps layer (in case of MNIST dataset), used for image classification prediction.

    Args:
        n_caps: The number of capsules in this layer.
        dims_caps: The dimension of the output vector of a capsule.
        r_iter: Number of routing iterations.
    """

    def __init__(self, n_caps, dims_caps, r_iter=3, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)

        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.r_iter = r_iter

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"n_caps": self.n_caps, "dims_caps": self.dims_caps, "r_iter": self.r_iter}
        )
        return config

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), "The input Tensor of a ClassCaps should have shape=[None, input_n_caps, input_dims_caps]"

        self.input_n_caps = input_shape[1]
        self.input_dims_caps = input_shape[2]

        # Transform matrix, from each input capsule to each output capsule, there's a unique weight as in Dense layer.
        self.init_sigma = 0.1
        self.W_init = tf.random.normal(
            shape=(
                1,
                self.input_n_caps,
                self.n_caps,
                self.dims_caps,
                self.input_dims_caps,
            ),
            stddev=self.init_sigma,
        )
        self.W = tf.Variable(self.W_init, name="W")

        self.built = True

    def call(self, inputs):
        # Replicate num_capsule dimension to prepare being multiplied by W
        input_caps_out_tile = tf.expand_dims(tf.expand_dims(inputs, -1), 2)
        input_caps_out_tiled = tf.tile(input_caps_out_tile, [1, 1, self.n_caps, 1, 1])

        # Matmul doesn't care about batch_size (it just uses the same self.W multiple times)
        inputs_hat = tf.matmul(self.W, input_caps_out_tiled)

        # ROUTING ALGORITHM
        raw_weights = tf.zeros([1, self.input_n_caps, self.n_caps, 1, 1])

        for i in range(self.r_iter):
            # Line 4, computes Eq.(3)
            routing_weights = tf.nn.softmax(raw_weights, axis=2)

            # Line 5, 6
            outputs = squash(
                tf.reduce_sum(
                    tf.multiply(routing_weights, inputs_hat), axis=1, keepdims=True
                ),
                axis=-2,
            )

            # Line 7
            if i < self.r_iter - 1:
                outputs_tiled = tf.tile(outputs, [1, self.input_n_caps, 1, 1, 1])
                raw_weights += tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)

        return tf.squeeze(outputs, axis=[1, -1]), routing_weights


def mask(inputs):
    """ Mask a Tensor with shape (batch_size, n_capsules, dim_vector).

    It can be done either by selecting the capsule with max length or by an additional input mask.
    The first is usually the method for testing, the second is the one for the training.

    Args:
        inputs: Either a tensor to be masked (output of the class capsules)
                or a tensor with both the tensor and an additional input mask
    """
    # Mask provided?
    if type(inputs) is tuple or type(inputs) is list:
        inputs, mask = inputs[0], inputs[1]
    else:
        # Calculate the mask by the max length of capsules.
        x = compute_vectors_length(inputs)
        # Generate one-hot encoded mask
        mask = one_hot(indices=argmax(x, 1), num_classes=x.get_shape().as_list()[-1])

    # Mask the inputs
    masked = batch_flatten(inputs * expand_dims(mask, -1))
    return masked


def compute_vectors_length(vecs, axis=-1):
    """
    Compute vectors length. This is used to compute final prediction as probabilities.

    Args:
        vecs: A tensor with shape (batch_size, n_vectors, dim_vector)

    Returns:
        A new tensor with shape (batch_size, n_vectors)
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vecs), axis) + epsilon())


def squash(vectors, axis=-1):
    """The non-linear activation used in Capsule, computes Eq.(1)

    It drives the length of a large vector to near 1 and small vector to 0.

    Args:
        vectors: The vectors to be squashed, N-dim tensor.
        axis: The axis to squash.
    Returns:
        A tensor with the same shape as input vectors, but squashed in 'vec_len' dimension.
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon())
    return scale * vectors
