from os.path import join as os_path_join

# Import TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks


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
        callbacks=[lr_decay, WeightsSaver(args.weights_save_dir, args.save_freq)],
    )

    # Save final weights at the end of the training
    model.save_weights(os_path_join(args.weights_save_dir, "trained.h5"))

    return model
