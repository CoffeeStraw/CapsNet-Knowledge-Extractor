"""
CapsuleNetwork TF 2.2 Implementation
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
Author: Antonio Strippoli
"""
# General imports
from os.path import join as os_path_join

# Import TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# Import CapsuleNetwork model for MNIST classification
from capsnet import CapsuleNet

# Import some utilities
from utils import parse_args, pickle_dump


def load_mnist():
    """Loads and prepares MNIST dataset.
    """
    from tensorflow.keras.datasets import mnist

    # Preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train.astype("float32"))
    y_test = to_categorical(y_test.astype("float32"))

    return (x_train, y_train), (x_test, y_test)


def train(model, data, args):
    """Perform training, saving weights at the end of all the epochs

    Args:
        model: The CapsuleNet model to train.
        data: The dataset that you want to train: ((x_train, y_train), (x_test, y_test))
        args: The arguments parsed at the beginning of the script.

    Returns:
        The trained model
    """
    # Unpack data
    (x_train, y_train), (x_test, y_test) = data

    # Define margin_loss function
    def margin_loss(y_true, y_pred):
        """Local function for margin loss, Eq.(4).

        When y_true[i, :] contains more than one `1`, this loss should work too (not tested).

        Arguments:
            y_true: Correct labels one-hot encoded (batch_size, n_classes)
            y_pred: Output of the DigitCaps Layer (batch_size, n_classes, n_capsules)
        Returns:
            A scalar loss value.
        """
        # Calculate loss
        L = y_true * tf.square(tf.maximum(0.0, 0.9 - y_pred)) + 0.5 * (
            1 - y_true
        ) * tf.square(tf.maximum(0.0, y_pred - 0.1))

        return tf.reduce_mean(tf.reduce_sum(L, 1))

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=args.lr),
        loss=[margin_loss, "mse"],
        loss_weights=[1.0, args.lam_recon],
        metrics=["accuracy"],
    )

    # Define a callback to reduce learning rate
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (args.lr_decay ** epoch)
    )

    # Define a callback that will save weights after every `args.save_freq` batches done
    class WeightsSaver(callbacks.Callback):
        def __init__(self, save_dir, save_freq):
            self.save_dir = save_dir
            self.save_freq = save_freq
            self.epoch = 1

        def on_epoch_end(self, epoch, logs={}):
            self.epoch += 1

        def on_batch_end(self, batch, logs={}):
            if batch % self.save_freq == 0:
                # Save model current state for later visualization
                save_name = os_path_join(self.save_dir, f"{self.epoch}-{batch}.h5")
                self.model.save_weights(save_name)

    # Simple training without data augmentation
    model.fit(
        x=(x_train, y_train),
        y=(y_train, x_train),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=((x_test, y_test), (y_test, x_test)),
        callbacks=[lr_decay, WeightsSaver(args.training_save_dir, args.save_freq)],
    )

    # Save final weights at the end of the training
    model.save_weights(os_path_join(args.save_dir, "trained.h5"))
    return model


def test(model, data, args):
    pass


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Set model args and save them for later model reinstantiation
    model_params = {
        "input_shape": x_train.shape[1:],
        "batch_size": args.batch_size,
        "n_class": y_train.shape[1],
        "r_iter": 3,
    }
    pickle_dump(model_params, os_path_join(args.save_dir, "model_params.pkl"))

    # Instantiate Capsule Network Model
    model, _ = CapsuleNet(**model_params)

    # Show a complete summary
    model.summary()

    # Train!
    model = train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
