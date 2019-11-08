import argparse
import os
from tensorFlow.train import TfTrainer
from mxNet.train import MxNetTrainer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="Framework to be used for training, e.g. tf (for TensorFlow) or mx (for mxnet)", default='tf')
    ap.add_argument('-m', '--mode', required=True, help='Specify whether you want to perform training or inference. e.g. -m inference', default='training')
    ap.add_argument('-t', '--type', required=True, help='The type of hemorrhage you would like to train the model for. e.g. -t epidural for type epidural', default='epidural')

    args = vars(ap.parse_args())

    if args['framework'] == 'tf':
        tfTrainer = TfTrainer(args['type'])
        tfTrainer.train()
    elif args['framework'] == 'mx':
        mxNetTrainer = MxNetTrainer(args['type'])
        if args['mode'] == 'training':
            mxNetTrainer.train()
        else:
            mxNetTrainer.inference()