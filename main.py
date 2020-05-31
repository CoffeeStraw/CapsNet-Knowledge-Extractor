"""
CapsuleNetwork TF 2.2 Implementation
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
Author: Antonio Strippoli
"""

# Import TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# Import CapsuleNetwork model for MNIST classification
from capsnet import CapsuleNet
from capslayers import compute_vectors_length

# Import some utilities
from utils import parse_args


def load_mnist():
    """Loads and prepares MNIST dataset.
    """
    from tensorflow.keras.datasets import mnist

    # Preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def train(model, data, args):
    """Perform training, saving weights at the end of all the epochs
    """

    # Unpack data
    (x_train, y_train), (x_test, y_test) = data

    def margin_loss(y_true, y_pred):
        """Local function for margin loss, Eq.(4).

        When y_true[i, :] contains more than one `1`, this loss should work too (not tested).

        Arguments:
            y_true: Correct labels one-hot encoded (batch_size, n_classes)
            y_pred: Output of the DigitCaps Layer (batch_size, n_classes, n_capsules)
        Returns:
            A scalar loss value.
        """
        # Calculate length of output vectors of DigitCaps
        y_pred = compute_vectors_length(y_pred)

        # Calculate loss
        L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

        return tf.reduce_mean(tf.reduce_sum(L, 1))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'output_1': 'accuracy'})

    # Training without data augmentation:
    model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_test, y_test])

    model.save_weights(args.save_dir + '/trained_model.h5')


def test(model, data, args):
    pass


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Instantiate Capsule Network Model
    model = CapsuleNet(n_class=10, r_iter=3)

    # Build and show summary
    model.build(input_shape=x_train.shape)
    model.summary()

    # Train!
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
