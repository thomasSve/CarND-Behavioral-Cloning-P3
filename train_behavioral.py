
import os
import os.path as osp
import sys
import numpy as np
from models import lenet, nvidia_cnn

def generator(data):
    pass

def train_model(batch = 64, epochs = 100, learning_reate = 0.001):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train the behavioral model')
    parser.add_argument('--batch', dest='batch',
                        help='batch size to train',
                        default=64, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to run over training set',
                        default=100, type=int)
    parser.add_argument('--l', dest='learning_rate',
                        help='set the learning rate of model',
                        default=0.001, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    Train behavioral model
    """

    args = parse_args()
    train_model()
    
