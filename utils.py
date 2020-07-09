"""
Utilities file, it contains functions not concerning Neural Networks.
It is usefull to separate logic.
Author: Antonio Strippoli
"""
import os
import argparse


def parse_args():
    # Getting/setting the hyper-parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    
    # General
    parser.add_argument('--save_dir', default='result', help="The directory that will contains every output of the execution.")
    parser.add_argument('--save_freq', default=100, type=int, help="The number of batches after which weights are saved.")

    # Testing or training?
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    # Initial weights?
    parser.add_argument('-w', '--weights', default=None,
                    help="The path of the saved weights. Should be specified when testing")

    # Training
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs for the training.")
    parser.add_argument('--batch_size', default=100, type=int, help="Size of the batch used for the training.")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    
    # Capsule Network
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")

    # Parse arguments from command line
    args = parser.parse_args()

    # Construct save dir path for training weights
    args.training_save_dir = os.path.join(args.save_dir, 'training')

    # Creating results directories, if they do not exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.training_save_dir):
        os.mkdir(args.training_save_dir)

    return args
