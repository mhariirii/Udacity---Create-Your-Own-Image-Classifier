import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
import json
from torch import nn
from torch import optim
import seaborn as sns
from PIL import Image
from collections import OrderedDict
import argparse
import utilities
import foldermodel

parser = argparse.ArgumentParser()
parser.add_argument('--img', default='flowers/test/100/image_07896.jpg', type = str)
parser.add_argument('--checkpoint', default = 'checkpoint.pth', type = str)
parser.add_argument('--top_k', default = 5, type = int)
parser.add_argument('--category_names', default = 'cat_to_name.json')

args = parser.parse_args()


top_k = args.top_k
image_path = args.img
checkpoint_path = args.checkpoint
labels = args.category_names

model = foldermodel.load_checkpoint(checkpoint_path)

with open(labels, 'r') as f:
    cat_to_name = json.load(f)


probs, classes = foldermodel.predict(image_path, model)
probs = probs.cpu()
probs = probs[0].detach().numpy()*100

                                             
labels = []
    
for each in classes:
    
    labels.append(cat_to_name[each])
        
for i in range(top_k):
    print("{} has a probability = {}%".format(labels[i],probs[i]))
        
print("prediction succeffully doen")
    
    
