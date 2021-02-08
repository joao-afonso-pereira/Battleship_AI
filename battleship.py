# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:44:40 2020

@author: João Afonso
"""

import numpy as np 
import random
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use("Agg")

# method used with the ndimage function that checks the neighbors of a boat cell 
def get_neighbors(values):
    return values.sum()

# class of the game
class Battleship():
    
    def __init__(self, save_path, boats = 3):
        
        self.num_boats = boats
        self.save_path = save_path

        self.create_boards()
            
    def create_boards(self):
        
        #the board starts as 2D array with 9 rows, 9 columns and all values set to zero
        self.main_board = np.zeros((9,9))
        
        #initialization of some variables of the game
        self.win = False
        self.hits = []
        self.attempts = []
        self.num_attempts = 0
        self.boats_destroyed = []
        self.score = 0
        
        #create the board, getting the list of boats and the list of blocked cells (not possible to insert a boat cell) 
        self.boat_list = []
        self.not_possible = []
        for _ in range(self.num_boats):
            self.boat_list.append(self.add_boat())
            rows, cols = np.where(self.main_board == -1)
            self.not_possible = [(rows[i], cols[i]) for i in range(len(rows))]
        
        #sort the cells of each boat
        for boat in self.boat_list:
            boat = boat.sort()
        
        #replace all -1 in the board by 0, so now a boat cell is represented by one and the rest is 0
        self.main_board = np.where(self.main_board!=1, 0, self.main_board)
        
        #create the board that will be presented to the player
        self.player_board = np.zeros_like(self.main_board)
            
        self.save_board_pic(self.main_board, self.save_path + '/secret_board.png')
        self.save_board_pic(self.player_board, self.save_path + '/player_board.png')
            
    def add_guess(self, guess, episode, step):
        
        self.num_attempts += 1
        
        guess = np.array(guess)
        
        twoD_action = guess.reshape(9,9)
        
        try:
            guess = np.where(twoD_action == guess[np.argmax(guess)])
        except: 
            guess = np.where(twoD_action == guess[0][np.argmax(guess)])
        
        #get the row and column of the guess (correspondent to the game board, not the player board)
        guess_row = guess[0][0]
        guess_col = guess[1][0]
        
        # guess_row = int(0 + (guess_row - 0)/(1 - 0) * (8 - 0))
        # guess_col = int(0 + (guess_col - 0)/(1 - 0) * (8 - 0))
        
        if guess_row < 0 or guess_row > 8 or guess_col < 0 or guess_col > 8:
            self.save_board_pic(self.player_board, self.save_path + '/player_board_ep{}_step{}.png'.format(episode, step))
            # self.score = - 1
            return (guess_row, guess_col), self.player_board, self.score, self.win
        
        #check if is repeated guess
        if (guess_row, guess_col) in self.attempts:
            self.save_board_pic(self.player_board, self.save_path + '/player_board_ep{}_step{}.png'.format(episode, step))
            # self.score = - 1
            return (guess_row, guess_col), self.player_board, self.score, self.win
        
        self.attempts.append((guess_row, guess_col))
        
        #check if the guess hit a boat
        is_boat = any((guess_row, guess_col) in sublist for sublist in self.boat_list) 
        if is_boat:       
            self.score = self.score + 1
            self.player_board[guess_row][guess_col] = "3"
            self.hits.append((guess_row, guess_col))
            self.check_boats_destroyed()
            
        else:
            # self.score = 0.5
            self.player_board[guess_row][guess_col] = "2"
            
            
        self.save_board_pic(self.player_board, self.save_path + '/player_board_ep{}_step{}.png'.format(episode, step))
            
        # self.score = 0.7 * (len(self.hits) / (self.num_boats*3)) + 0.2 * (len(self.hits) / len(self.attempts)) + 0.1 * (len(self.attempts) / self.num_attempts)

        #check if game was won
        self.win = self.check_win()

        return (guess_row, guess_col), self.player_board, self.score, self.win
    
    def check_boats_destroyed(self):
        
        #check for destroyed boats
        
        for i in range(len(self.boat_list)):
            if all(i in self.hits for i in self.boat_list[i]) and i not in self.boats_destroyed:
                self.boats_destroyed.append(i)
                #print(">> {} boat(s) destroyed!".format(len(self.boats_destroyed)))
        
    def check_win(self):
        
        #check if game was won
        
        win = False
        if self.num_boats == len(self.boats_destroyed):
            win = True
            
        return win
    
    def get_win(self):
        return self.win
    
    def get_attempts(self):
        return self.attempts
    
    def get_num_attempts(self):
        return self.num_attempts
    
    def save_board_pic(self, board, path):
        
        #create image and save it as png
        
        colors = {   0:  [90,  155,  255],
                     1:  [88,  88,  88],
                     2:  [14,  11,  167],
                     3:  [255,  0,  0]}
        
        image = np.array([[colors[val] for val in row] for row in board], dtype='B')

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        ax.set_xticks([0,1,2,3,4,5,6,7,8])
        ax.set_yticks([0,1,2,3,4,5,6,7,8])
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.imshow(image)
        ax.set_yticklabels(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        for i in list(np.arange(0.5, 8.5, 1)):
            plt.axvline(x = i, color = 'black', linestyle = '-', lw=0.5) 
            plt.axhline(y = i, color = 'black', linestyle = '-', lw=0.5) 
        plt.tick_params(axis='both', labelsize=12, length = 0)
        plt.title("Score = {}".format(self.score))
        plt.savefig(path)
        plt.close('all')
               
    def get_player_board(self):     
        return self.player_board
    

    def add_boat(self):
        
        #add a 3 cell boat to the board
        
        done = False
        while done == False:
            
            #create temporary board
            board = np.zeros((9,9))
            
            boat = []
            
            #check if initial boat cell is valid
            invalid = True
            while invalid:
                row = random.randint(0, 8)
                col = random.randint(0, 8)
                if self.main_board[row, col] == 0:
                    invalid = False
            
            #get initial cell
            board[row, col] = 1
            boat.append((row, col))
            
            #possible second cell
            possible = [(boat[-1][0]-1, boat[-1][1]), (boat[-1][0]+1, boat[-1][1]), (boat[-1][0], boat[-1][1]-1), (boat[-1][0], boat[-1][1]+1)]
            
            #remove the invalid second cells
            possible = [coordinates for coordinates in possible if -1 not in coordinates and self.main_board.shape[0] not in coordinates and coordinates not in self.not_possible]
            
            #get the random second cell
            row, col = random.sample(possible, 1)[0]
            board[row, col] = 1
            boat.append((row, col))
            
            #possible third cell
            if boat[-1][0] == boat[0][0]:
                if boat[-1][1] > boat[0][1]:
                    possible = [(boat[-1][0], boat[0][1]-1), (boat[-1][0], boat[-1][1]+1)]
                else:
                    possible = [(boat[-1][0], boat[0][1]+1), (boat[-1][0], boat[-1][1]-1)]
            else:
                if boat[-1][0] > boat[0][0]:
                    possible = [(boat[0][0]-1, boat[-1][1]), (boat[-1][0]+1, boat[-1][1])]
                else:
                    possible = [(boat[0][0]+1, boat[-1][1]), (boat[-1][0]-1, boat[-1][1])]
            
            #validate possibilities
            possible = [coordinates for coordinates in possible if -1 not in coordinates and self.main_board.shape[0] not in coordinates and coordinates not in boat and coordinates not in self.not_possible]
            
            #add third cell
            try:
                row, col = random.sample(possible, 1)[0]
            except:
                continue
            board[row, col] = 1    
            boat.append((row, col))
        
            #get neighbors of the boat
            footprint = np.array([[1,1,1],
                                  [1,0,1],
                                  [1,1,1]])
            
            board = ndimage.generic_filter(board, get_neighbors, footprint=footprint)
            
            #define the neighbors and boats as -1
            board = np.where(board!=0, -1, board)
            
            #define boat cells as 1
            for row, col in boat:
                board[row, col] = 1
                
            #join the temporary board to the main board
            self.main_board = self.main_board + board
            
            done = True
    
        return boat
    
def main():   
    
    game = Battleship(boats=3)  

if __name__ == '__main__':
    
    main()



