# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from battleship import Battleship
import sys
import pandas as pd
import shutil
import os

def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 3.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return "random", random.sample(range(-100, 100), 81)
        act_values = self.model.predict(state)
        return "model", act_values  # returns action

    def replay(self, batch_size):

        counter = 0
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            counter +=1
            
            sys.stdout.write('\r'+ '     - Training model ... iteration {}/{}'.format(counter, batch_size))
            sys.stdout.flush()   
            
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                
            target_f = self.model.predict(state)

            target_f[0][np.argmax(target_f[0])] = target
    
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    
    EPISODES = 1000
    
    state_size = 81
    action_size = 81
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 256
    
    attempts = []
    ships_destroyed = []
    final_score = []

    for e in range(EPISODES):
        
        path = "figs/episode{}".format(e)
        if not os.path.exists(path):
            os.mkdir(path)
        
        go=False
        while not go:
            try:
                env = Battleship(path, 4)
                go = True
            except:
                pass
            
        state = env.get_player_board()
        state = state.flatten() 
        state = np.reshape(state, [1, state_size])
        
        action_list = []
        rewards_list = []
        who_list = []
        eps_list = []
        
        print("> Episode {}/{}".format(e, EPISODES))

        for time in range(120):
            
            state = state.flatten() 
            state = np.reshape(state, [1, state_size])
 
            who, action = agent.act(state)
            
            who_list.append(who)

            guess, next_state, reward, done = env.add_guess(action, e, time)
            
            sys.stdout.write('\r'+ '     - Guess = {} | Reward = {}'.format(guess, reward))
            sys.stdout.flush()
            
            reward = reward if not done else 10
            
            action_list.append(guess)
            
            rewards_list.append(reward)
            
            # reward = reward if not done else 10
            next_state = next_state.flatten() 
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            
            eps_list.append(agent.epsilon)
            
            if done:
                break
                
            if len(agent.memory) > batch_size: 
                
                agent.replay(batch_size)
                
        print("\n     - Score = {} | Won = {} | Epsilon = {:.2}".format(reward, done, agent.epsilon))
                
        ep_df = pd.DataFrame({'step': list(range(time+1)), 'action': action_list, 'who': who_list, 'reward': rewards_list, 'epsilon': eps_list}) 
        ep_df.to_excel("episodes/episode_{}.xlsx".format(e), index=False)          
                
        attempts.append(time)
        ships_destroyed.append(env.boats_destroyed)
        final_score.append(env.score)
        
        make_archive(path, path + ".zip")
        shutil.rmtree(path)
        
    df = pd.DataFrame({'episode': list(range(EPISODES)), 'attempts': attempts, 'ships_destroyed': ships_destroyed, 'final_score': final_score})
    df.to_excel("episodes_summary.xlsx", index=False)            
            