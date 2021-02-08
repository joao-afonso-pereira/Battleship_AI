# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:33:42 2021

@author: Asus
"""

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from statistics import mean
import pickle
import math
import sys

from mlp import MLP
from battleship import Battleship

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

#%% 

EPOCHS = 10

loss_fn = F.cross_entropy

LEARNING_RATE = 1e-04
REG = 1e-04

def fit(model, device, num_boats=3):

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REG)

    train_history = []
    
    y = np.array([num_boats*3]*EPOCHS)
    y = torch.from_numpy(y)
    y = y.type(torch.LongTensor)

    for epoch in range(EPOCHS):    
        
        print("EPOCH {}/{}".format(epoch+1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        
        game = Battleship(boats=num_boats)  
        
        while game.get_win()==False:
            
            board = game.get_player_board()
            
            
            board = torch.from_numpy(board)
            board = board.type(torch.LongTensor)
            board = torch.reshape(board, (1, 1, 9, 9))
            
            board = board.float()
            
            board = board.to(device)
            
            guess = model(board)
            
            print(guess)
            
            #add guess
            game.add_guess(guess)
            
            game.save_board_pic(game.get_player_board(), 'figs/player_board_epoch_{}_attempt_{}.png'.format(epoch, game.get_num_attempts()))
        
        attempts = game.get_attempts()
        
        num_attempts = game.get_num_attempts()
        
        loss = loss_fn(num_attempts, y[epoch])
        
        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step() 
        
        train_history.append(attempts)
        
        with open("attempts_epoch_{}.txt".format(epoch+1),"w") as file:
            
            for a in attempts:
                file.write(a)
                file.write("\n") 
        
        # display the training loss
        print()
        print(">> Attempts = {} | Loss = {}".format(attempts, loss))


    return model, train_history

def main():

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU")
    
    model = MLP().to(DEVICE)
    model.apply(weights_init_uniform)
        
    fit(model, DEVICE)

if __name__ == '__main__':
    main()