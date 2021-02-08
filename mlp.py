from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import mean
import sys
from torchsummary import summary

class Flat(nn.Module):
    
    def __init__(self):
        super(Flat, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class MLP(nn.Module):
    
    def __init__(self):
        
        super(MLP, self).__init__()
        
        self.main = nn.Sequential(
            
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            Flat(),

            nn.Linear(400, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, 2),
            nn.Sigmoid()
            
        )

    def forward(self, input):

        return self.main(input)

  
if __name__ == '__main__':

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU...")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU...")
    print()

    
    model = MLP().to(DEVICE)
    print(model)

    summary(model, (1, 9, 9))