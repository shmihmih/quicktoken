#!/usr/bin/env python
# coding: utf-8

# # Agent

# In[2]:


import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.firstIter = True

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=200, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=100, activation="relu"))
        model.add(Dense(units=100, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.01))

        return model

    def act(self, state):
        rand_val = np.random.rand()
        if not self.is_eval and rand_val <= self.epsilon:
            return random.randrange(self.action_size)

        if(self.firstIter):
            self.firstIter = False
            return 1
        options = self.model.predict(state)
        #print("Using prediction",options)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        #mini_batch = []
        #l = len(self.memory)
        #for i in range(l - batch_size + 1, l):
        #    mini_batch.append(self.memory[i])
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 


# # Functions 

# In[3]:


import numpy as np
import math

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])


# In[4]:


#from agent.agent import Agent
#from functions import *
#import sys

#if len(sys.argv) != 4:
#    print("Usage: python train.py [stock] [window] [episodes]")
#    exit()

#model_name = 'model_ep50'


#stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stock_name, window_size, episode_count = '^GSPC_2011', 20, 200
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = 100*(data[t] - bought_price)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        
        if done ==True and reward==0:
            reward=-500
        
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))


# In[5]:


import keras
from keras.models import load_model


stock_name, model_name = '^GSPC_2011','model_ep50'
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []


for t in range(l):
   
    action = agent.act(state)
    print(action)
    # sit
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # buy
        agent.inventory.append(data[t])
        print ("Buy: " + formatPrice(data[t]))

    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        reward = 100*(data[t] - bought_price)
        total_profit += data[t] - bought_price
        #print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    done = True if t == l - 1 else False
    
    if done ==True and reward==0:
        reward=-500
    
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print ("--------------------------------")
        print (stock_name + " Total Profit: " + formatPrice(total_profit))
        print ("--------------------------------")


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,10))
plt.plot(getStockDataVec(stock_name))


# In[39]:


agent.memory


# In[ ]:




