"""
Usage:

Predict raw categories:
python predict.py "~/.torch/datasets/flowers\test\1\image_06743.jpg" "models\ckp_Flowers_vgg13_512_256_0.0001_20_best.pth"  --gpu

Predict and convert categories to names using a json dictionary:
python predict.py "~/.torch/datasets/flowers\test\1\image_06743.jpg" "models\ckp_Flowers_vgg13_512_256_0.0001_20_best.pth"  --gpu --category_names "~/.torch/datasets/flowers/cat_to_name.json"
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
        description='Predit class for an input image from a trained checkpoint'
    )

    parser.add_argument('img_file',
                        help='Path to image')

    parser.add_argument('ckp_file',
                        help='Path to checkpoint')

    parser.add_argument('--category_names', action='store',
                        dest='category_names',
                        help='Path to category names (json file)')

    parser.add_argument('--top_k', action='store',
                        dest='top_k', type=int, default=2,
                        help='K most likely classes')

    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', default=False,
                        help='Predict using CUDA:0 if available')

    results = parser.parse_args()

    return results

if __name__ == "__main__":
    # Get cmd args
    args = parse_input()
    
    # Instanciate Image Classifier Class
    ic = ImageClassifier()

    # Request GPU if available
    ic.use_gpu(args.gpu)

    # Load class to names
    if args.category_names is not None and not ic.load_class_names(args.category_names):
        exit()

    # Load checkpoint to predict
    if not ic.load_checkpoint(args.ckp_file):
        exit()
    
    # Predict the class for the provided image path
    ic.print_predictions(ic.predict(args.img_file, args.top_k, show_image=True))