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
from tensorflow.keras.losses import binary_crossentropy

from collections import deque
import random

# Load environment
env = or_gym.make('Knapsack-v2')
env.mask = False

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 5_000
test_episodes = 100


def agent(item_count = env.N):
    """ 
    Requires the values of all the available items
    """
    model = Sequential()
    model.add(Dense(item_count * 2, activation="relu", input_shape=(item_count,)))
    model.add(Dense(item_count * 2, activation="relu"))
    model.add(Dense(item_count, activation="sigmoid"))
    model.compile(Adam(1e-3), binary_crossentropy, metrics=["accuracy"])
    
    return model

def train(env, replay_memory, model, target_model, done):
    """
    Train model with Belman equation
    """
    learning_rate = 0.2 # Learning rate
    discount_factor = 0.8 #0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 50 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    #print(mini_batch)
    #print("Len minibatch", len(mini_batch))
    current_states = np.array([transition[0] for transition in mini_batch])
    #print(current_states.shape)
    #print(current_states)
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index]) # Future Q value
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q # Belman equation

        X.append(observation)
        Y.append(current_qs)
        
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True) # Update neural network


def avail_item_values():
    """
    Returns the available item values
    """
    avail_item_values = []
    for item in range(env.N):
        if env.item_weights[item] + env.current_weight < env.max_weight:
            if env.item_limits[item] > 0:
                avail_item_values.append(env.item_values[item] / env.item_weights[item])
            else:
                avail_item_values.append(0)
        else:
            avail_item_values.append(0)
    
    return np.array(avail_item_values)

def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.0004 # How fast the epsilon decays

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent()
    # Target Model (updated every 100 steps)
    target_model = agent()
    # Set the same weights as the Main Model for Target Model
    target_model.set_weights(model.get_weights())
    # Initialize Replay Memory
    replay_memory = deque(maxlen=10_000)

    steps_to_update_target_model = 0
        
    for episode in range(train_episodes):
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
                # Exploit best known action
                predicted = model.predict(observation.reshape(1, env.N))
                action = np.argmax(predicted)
                
            new_observation, reward, done, info = env.step(action)
            new_observation = avail_item_values()
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)
                #print("weights\n", model.get_weights())

            observation = new_observation
            total_training_rewards += reward
            reward_list.append(reward); item_list.append(action)

            if done:
                print(u'_________ n = {} _________'
                      u'\nTotal training rewards: {}\n'
                      u'Reward list = {}\n'
                      u'item list {}\n'.format(episode,total_training_rewards,reward_list, item_list))

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        print("Epsilon:", epsilon, "\n")
    env.close()

if __name__ == '__main__':
    main()