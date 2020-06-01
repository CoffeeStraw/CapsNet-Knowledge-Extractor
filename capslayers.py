import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, Reshape, Lambda
from tensorflow.keras.backend import epsilon
from tensorflow.keras import initializers


class PrimaryCaps(Layer):
    """
    A PrimaryCaps layer. More info To Be Added.
    """

    def __init__(self, n_capsules, out_dim_capsule, kernel_size, strides, padding, name="primary_caps"):
        super(PrimaryCaps, self).__init__(name=name)

        # Apply Convolution n_capsules times
        self.conv2d = Conv2D(filters=n_capsules*out_dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding,
                             name='primarycaps_conv2d')

        # Reshape the convolutional layer output
        self.reshape = Reshape(
            target_shape=[-1, out_dim_capsule], name='primarycaps_reshape')

        # Squash the vectors output
        self.squash = Lambda(_squash, name='primarycaps_squash')

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.reshape(x)
        return self.squash(x)


class ClassCaps(Layer):
    """A DigitCaps layer (in case of MNIST dataset), used for image classification prediction.

    Args:
        n_capsules: The number of capsules in this layer.
        out_dim_capsule: The dimension of the output vector of a capsule.
        r_iter: Number of routing iterations.
    """

    def __init__(self, n_capsules, out_dim_capsule, r_iter=3, name="digit_caps"):
        super(ClassCaps, self).__init__(name=name)

        self.n_capsules = n_capsules
        self.out_dim_capsule = out_dim_capsule
        self.r_iter = r_iter

    def build(self, input_shape):
        assert len(
            input_shape) >= 3, "The input Tensor should have shape=[None, input_n_capsules, input_out_dim_capsule]"
        self.input_n_capsules = input_shape[1]
        self.input_out_dim_capsule = input_shape[2]

        # Transform matrix, from each input capsule to each output capsule, there's a unique weight as in Dense layer.
        self.W = self.add_weight(shape=[self.n_capsules, self.input_n_capsules,
                                        self.out_dim_capsule, self.input_out_dim_capsule],
                                 initializer=initializers.get(
                                     'glorot_uniform'),
                                 name='W')

    def call(self, inputs):
        # Prepare input to be multiplied by W
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        # Replicate num_capsule dimension
        inputs_tiled = tf.tile(inputs_expand, [1, self.n_capsules, 1, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        inputs_hat = tf.squeeze(
            tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))

        # ROUTING ALGORITHM
        # The prior for coupling coefficient, initialized as zeros
        b = tf.zeros(
            shape=[inputs.shape[0], self.n_capsules, 1, self.input_n_capsules])

        for i in range(self.r_iter):
            # Line 4, computes Eq.(3)
            c = tf.nn.softmax(b, axis=1)

            # Line 5 and 6
            outputs = _squash(tf.matmul(c, inputs_hat))  # [None, 10, 1, 16]

            # Line 7, but avoiding calcs on the last iteration
            if i < self.r_iter - 1:
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)

        return tf.squeeze(outputs)


def compute_vectors_length(vecs):
    """
    Compute vectors length. This is used to compute final prediction as probabilities.

    Args:
        vecs: A tensor with shape (batch_size, n_vectors, dim_vector)

    Returns:
        A new tensor with shape (batch_size, n_vectors)
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vecs), -1) + epsilon())


def _squash(vectors, axis=-1):
    """The non-linear activation used in Capsule, computes Eq.(1)

    It drives the length of a large vector to near 1 and small vector to 0.

    Args:
        vectors: The vectors to be squashed, N-dim tensor.
        axis: The axis to squash.
    Returns:
        A tensor with the same shape as input vectors, but squashed in 'vec_len' dimension.
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / \
        tf.sqrt(s_squared_norm + epsilon())
    return scale * vectors
