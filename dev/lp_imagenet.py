import numpy as np
import pandas as pd

from time import time
import scipy.stats as stat

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet,

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import pickle

from utils import nn_score, input_transformer_gaussian
import tensorflow as tf
from math import ceil
from math import floor
import os


LOG_DIR = "/home/karim-tito/sampling-reliability-measure/logs/imagenet/"
DATA_PATH = "/home/karim-tito/sampling-reliability-measure/data/imagenet/"
dataset = "mnist"
NETS_DIR ="/home/karim-tito/sampling-reliability-measure/data/test_nets/"
DIM = 3*224*224
gaussian_gen = lambda N: np.random.normal(size =(N,DIM))
print(tf.config.list_physical_devices('GPU')[0])






