# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json

test = []
for filename in os.listdir(r"/home/wfy/code/jinnan_2/jinnan2_round2_test_b_20190424"):
    test.append(filename)

model = torch.load('new_model_round2_50.pkl',map_location='cpu')
#model = model.module
model.eval()
input_size = 224
from PIL import Image
transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_dict = {}
N=0
for i in test:
    print(i)
    path = os.path.join("/home/wfy/code/jinnan_2/jinnan2_round2_test_b_20190424/"+i)
    img = Image.open(path).convert('RGB')


    img = transform(img)
    img = torch.unsqueeze(img, 0)


    out = model(img)
    _, preds = torch.max(out, 1)    
    test_dict[i] = int(preds.numpy())   
    if preds==1:
        N=N+1
print(N)
with open("new_test_dict_b.json","w") as f:
    json.dump(test_dict,f)