"""
Usage:

Test an existing model:
python test.py "~/.torch/datasets/flowers" "models\ckp_Flowers_vgg13_512_256_0.0001_20_best.pth"  --gpu
"""

import argparse
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Local imports
from ImageClassifier import ImageClassifier

def parse_input():
    parser = argparse.ArgumentParser(
        description='Test an image classifier'
    )
    
    parser.add_argument('data_dir',
                        help='Path to dataset to test')

    parser.add_argument('ckp_file',
                        help='Path to checkpoint')


    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', default=False,
                        help='Test using CUDA:0')

    results = parser.parse_args()
    return results

if __name__ == "__main__":
    # Get cmd args
    args = parse_input()
    
    # Instanciate Image Classifier Class
    ic = ImageClassifier()

    # Request GPU if available
    ic.use_gpu(args.gpu)

    # Load Dataset
    if not ic.load_data(args.data_dir):
        exit()        
          
    # Load checkpoint to resume training
    if not ic.load_checkpoint(args.ckp_file):
        exit()

    # Test the dataset
    ic.test()