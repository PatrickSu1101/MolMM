#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from argparse import ArgumentParser

from MolMM import *

seeds=42
torch.manual_seed(seeds)
np.random.seed(seeds)
torch.cuda.manual_seed_all(seeds)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = True

parser = ArgumentParser()
parser.add_argument("--name", '-n', type=str, default='CNN_pH')
parser.add_argument("--dir", '-d', type=str, default='results/TMP')
parser.add_argument("--ckpt_dir_1", '-c1', type=str, default='ckpt/TMP')
parser.add_argument("--loop", '-l', type=int, default=10)
parser.add_argument('--bar', '-b', type=bool, default=True)
parser.add_argument('--task_list', '-t', type=str, nargs='+', default=['logp','cls_permeability'])

args = parser.parse_args()
for num in range(args.loop):
    get_models(vars(args))
