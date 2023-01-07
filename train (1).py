import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import json
import torch.nn.functional as F
from torch import nn
from torch import optim
import seaborn as sns
from PIL import Image
from collections import OrderedDict
import argparse
import utilities
import foldermodel

parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", help="add data directory", default="flowers")
parser.add_argument("--arch", default="vgg19", type=str)
parser.add_argument("--learning_rate", default = 0.002)
parser.add_argument("--hidden_units", default=2048)
parser.add_argument("--epochs", default=8, type=int)
parser.add_argument("--save_dir", default="checkpoint.pth")
args = parser.parse_args()

data_dir = args.data_directory
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
save_dir = args.save_dir

trainloader, validloader, testloader, train_data = utilities.load_data(data_dir)
model, criterion, optimizer = foldermodel.model_set(arch, learning_rate, hidden_units)

foldermodel.train_model(model, trainloader,validloader, epochs, criterion, optimizer)

foldermodel.saving_checkpoint(model, save_dir, hidden_units, optimizer, train_data,)

print("model was successfully trained")