# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:41:18 2021

@author: 20172458
"""

import or_gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import RandomNormal
import pandas as pd

from collections import deque
import matplotlib.pyplot as plt

# Load environment
env = or_gym.make('Knapsack-v2')
env.mask = False

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

#___ PARAMETERS ___#
TRAIN_EPSISODES = 1000
TEST_EPSISODES = 100
LEARNING_RATE = 0.05 
DISCOUNT_RATE = 0.9 
MIN_REPLAY_SIZE = 250
BATCH_SIZE = 100 # Number of items to be sampled
FACTOR = 2 # How much one wants to punish or prize high rewards
HIGH_PER = 10 # The high and low percentage of rewards in the replay memory
MAX_EPSILON = 1 # You can't explore more than 100% of the time
MIN_EPSILON = 0.01 # At a minimum, we'll always explore 1% of the time
DECAY = 0.01 # How fast the epsilon decays
#_________________#

def agent(item_count = env.N):
    """ 
    Requires the values of all the available items
    """
    init = RandomNormal(mean=0.0, stddev=0.05, seed=10)
    model = Sequential()
    model.add(Dense(item_count * 3, activation="relu", input_shape=(item_count,), kernel_initializer = init))
    model.add(Dense(item_count * 2, activation="relu", kernel_initializer = init))
    model.add(Dense(item_count, activation="relu", kernel_initializer = init))
    model.add(Dense(item_count, kernel_initializer = init))
    model.compile(Adam(1e-3), MeanSquaredError(), metrics=["accuracy"])
    
    return model

def train(env, replay_memory, model, target_model, done, LEARNING_RATE, DISCOUNT_RATE):
    """
    Train model with Belman equation
    """
    len_replay = len(replay_memory)
    if len_replay < MIN_REPLAY_SIZE:
        return

    rewards = [transition[2] for transition in replay_memory] # Extract rewards from replay memory
    rewards.sort() # Sort the rewards list  
    nr_items = int(len_replay / HIGH_PER) # Number of items in the 'HIGH_PER' range
    high_list = [ x * (((i * FACTOR) + nr_items) / nr_items) for x, i in zip(rewards[-nr_items:], range(nr_items))] # Apply factor to high rewards 
    low_list = [ x * (i / (nr_items * FACTOR)) for x, i in zip(rewards[:nr_items], range(nr_items))] # Apply factor to low rewards
    rewards = low_list + rewards[nr_items:-nr_items] + high_list # Overwrite with the adjusted rewards list
    mini_batch_ind = np.random.choice(list(range(len(rewards))), BATCH_SIZE, p = [x/sum(rewards) for x in rewards]) # Sample the indices of observations
    # where the observations with high rewards are more likely sampled.
    mini_batch = [replay_memory[ind] for ind in mini_batch_ind] # Retrieve the observations with the given indices 
    current_states = np.array([transition[0] for transition in mini_batch]) # Extract the observations from the mini_batch
    current_qs_list = model.predict(current_states) 
    new_current_states = np.array([transition[3] for transition in mini_batch]) # Extract the new observation from the mini_batch
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + DISCOUNT_RATE * np.max(future_qs_list[index]) # Future Q value
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - LEARNING_RATE) * current_qs[action] + LEARNING_RATE * max_future_q # Belman equation

        X.append(observation)
        Y.append(current_qs)
        
    model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=True) # Update neural network


def avail_item_values():
    """
    Returns the available item values
    """
    avail_item_values = []
    for item in range(env.N):
        if env.item_weights[item] + env.current_weight < env.max_weight:
            if env.item_limits[item] > 0:
                avail_item_values.append(env.item_values[item])
            else:
                avail_item_values.append(0)
        else:
            avail_item_values.append(0)
    
    return np.array(avail_item_values)

def main(Decay, Learning_rate, Discount_rate):
    
    DECAY = Decay
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent()
    # Target Model (updated every 100 steps)
    target_model = agent()
    # Set the same weights as the Main Model for Target Model
    target_model.set_weights(model.get_weights())
    # Initialize Replay Memory
    replay_memory = deque(maxlen=800)
    
    episode_reward_history = []
    total_reward_history = []
    one = 0
    solved = 0

    steps_to_update_target_model = 0
        
    for episode in range(TRAIN_EPSISODES):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        reward_list = []; item_list = []
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            observation = avail_item_values()
            if random_number <= epsilon: # Explore
                action = env.action_space.sample() 
            else: # Exploit
                predicted = model.predict(observation.reshape(1, env.N))
                action = np.argmax(predicted)
                
            new_observation, reward, done, info = env.step(action)
            new_observation = avail_item_values()
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done, LEARNING_RATE, DISCOUNT_RATE)

            observation = new_observation
            total_training_rewards += reward
            reward_list.append(reward); item_list.append(action)

            if done:
                # print(u'_________ n = {} _________'
                #       u'\nTotal training rewards: {}\n'
                #       u'Reward list = {}\n'
                #       u'item list = {}\n'.format(episode,total_training_rewards,reward_list, item_list))

                if steps_to_update_target_model >= 100:
                   #print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)
        print("Epsilon:", epsilon, "\n")
        
        episode_reward_history.append(total_training_rewards)
        
        if len(episode_reward_history) > 100:
             del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        total_reward_history.append(running_reward)
        print("The average reward over the last 100 episodes is", running_reward)


        if running_reward > 1600:  # Condition to consider the task 
            if one == 0:
                solved = episode   
                one = 1
        #print("Solved at episode {}!".format(episode))
            
            
        
        env.close()
    return total_reward_history, solved

def line_plot(data, title = "", subtitle = "", ylabel = "", file_name = "output", line_thick = 1):
    """
    LINEPLOTTER
    
    data: must be of type list, multilist or Pandas DataFrame
    title: Title on the top of the figure (type = str)
    subtitle: Extra title (type = str)
    ylabel: Name of the ylabel (type = str)
    file_name: name of the file, saved in your current directory (type = str)
    line_thick: thickness of the lines
    """
    plt.figure(figsize=(15, 7)) # Figure size
    other_font = {'fontname': 'arial'} # Font type

    if isinstance(data, list):
        if isinstance(data[0], list): # Multilist
            i = 1
            for output in data:
                plt.plot(range(len(output)), output, linewidth = line_thick, label = "Scenario " + str(i))
                plt.scatter(range(len(output)), output, s = 8 * line_thick)
                i += 1
        else: # List
            plt.plot(range(len(data)), data, linewidth = line_thick)
            plt.scatter(range(len(data)), data, s = 5 * line_thick)

    elif isinstance(data, pd.DataFrame): # Pandas DataFrame
        col_names = list(data.columns)
        for name in col_names:

            plt.plot(range(len(data)), data[name], label = str(name), linewidth = line_thick)
            plt.scatter(range(len(data)), data[name], s = 5 * line_thick)

    else:
        print("{} are not allowed, only lists, multilists, and pandas DataFrames are allowed".format(type(data)))
    
    # Styling
    plt.ylabel(ylabel, fontsize = 16, ** other_font)
    plt.xlabel("Generations", fontsize = 16, ** other_font)
    plt.title(title, loc='left', fontsize=20, fontweight = "bold", ** other_font)
    plt.title(subtitle, loc='right', fontsize=14, color='dimgray', ** other_font)
    plt.legend(loc = 1, edgecolor = 'white')
    plt.savefig(file_name + ".jpg")
    plt.show()



def DecayRate():

    a = main(0.0001, LEARNING_RATE, DISCOUNT_RATE)
    b = main(0.0005, LEARNING_RATE, DISCOUNT_RATE)
    c = main(0.001, LEARNING_RATE, DISCOUNT_RATE)
    d = main(0.01, LEARNING_RATE, DISCOUNT_RATE)
    
    total = {"Decay = 0.0001": a, "Decay = 0.0005":b, "Decay = 0.001":c, "Decay = 0.01":d}
    df = pd.DataFrame(total)
    #total = [a,b,c,d]
    line_plot(df, "Comparison for different values of the decay rate", ylabel = "Average reward over last 100 episodes")

def LearningRate():
    
    a =  main(DECAY, 0.05, DISCOUNT_RATE)
    b =  main(DECAY, 0.1, DISCOUNT_RATE)
    c =  main(DECAY, 0.2, DISCOUNT_RATE)
    d =  main(DECAY, 0.4, DISCOUNT_RATE)
    
    total = {"Learning rate = 0.05": a, "Learning rate = 0.1":b, "Learning rate = 0.2":c, "Learning rate = 0.4":d}
    df = pd.DataFrame(total)
    
    #total = [a,b,c,d]
    line_plot(df, "Comparison for different values of the learing rate", ylabel = "Average reward over last 100 episodes")

def DiscountRate():
    
    a =  main(DECAY, LEARNING_RATE, 0.99)
    b =  main(DECAY, LEARNING_RATE, 0.95)
    c =  main(DECAY, LEARNING_RATE, 0.9)
    d =  main(DECAY, LEARNING_RATE, 0.8)
    
    total = {"Discount rate = 0.99": a, "Discount rate = 0.95":b, "Discount rate = 0.9":c, "Discount rate = 0.8":d}
    df = pd.DataFrame(total)
    
    #total = [a,b,c,d]
    line_plot(df, "Comparison for different values of the Discount rate", ylabel = "Average reward over last 100 episodes")

def Optimal():
    a = main(DECAY, LEARNING_RATE, DISCOUNT_RATE)
    line_plot(a[0], "Rewards with optimal parameters", ylabel = "Average reward over last 100 episodes")
    return a


#DecayRate()
#LearningRate()
#DiscountRate()  
a = Optimal()
print(a[1])

#if __name__ == '__main__':
#    main(DECAY, LEARNING_RATE, DISCOUNT_RATE)