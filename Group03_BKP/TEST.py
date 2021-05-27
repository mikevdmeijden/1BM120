# -*- coding: utf-8 -*-

import or_gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import RandomNormal

from collections import deque

FILE_LOC = './trained_model/saved_weights' # Location saved weights
NAAM_TEST = "JORICK"
VERSION = "v1"
LEARNING_RATE = 0.4 
DISCOUNT_RATE = 0.9 
MIN_REPLAY_SIZE = 250
BATCH_SIZE = 100 # Number of items to be sampled
FACTOR = 2 # How much one wants to punish or prize high rewards
HIGH_PER = 10 # The high and low percentage of rewards in the replay memory
TRAIN_EPSISODES = 1_000
TEST_EPSISODES = 100
MAX_EPSILON = 1 # You can't explore more than 100% of the time
MIN_EPSILON = 0.01 # At a minimum, we'll always explore 1% of the time
DECAY = 0.0005 # How fast the epsilon decays

env = or_gym.make('Knapsack-v2')
env.mask = False

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

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

def train_model(env, replay_memory, model, target_model, done):
    """
    Train model with Belman equation
    """
    global HIGH_PER, FACTOR, BATCH_SIZE, DISCOUNT_RATE, LEARNING_RATE, MIN_REPLAY_SIZE
    
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

def train_dqn():
    
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

    steps_to_update_target_model = 0
    all_rewards = []
        
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
                train_model(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward
            reward_list.append(reward); item_list.append(action)

            if done:
                print(u'_________ n = {} _________'
                      u'\nTotal training rewards: {}\n'
                      u'Reward list = {}\n'
                      u'item list = {}\n'.format(episode,total_training_rewards,reward_list, item_list))

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        
        all_rewards.append(reward)
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)
        print("Epsilon:", epsilon, "\n")
    
    env.close()
    model.save_weights(FILE_LOC + NAAM_TEST + "_" + VERSION)
    
    return all_rewards

def test_dqn(env):
    
    # 1. Initialize the model with saved weights
    try:
        model = agent(env.N)
        model.load_weights(FILE_LOC + NAAM_TEST + "_" + VERSION)
        
        all_rewards = []
            
        for episode in range(TEST_EPSISODES):
            observation = env.reset()
            observation = avail_item_values(env)
            done = False
            reward_list = []; item_list = []
            
            rewards = 0
            while not done:
                predicted = model.predict(observation.reshape(1, env.N))
                action = np.argmax(predicted)
                    
                new_observation, reward, done, info = env.step(action)
                new_observation = avail_item_values(env)
    
                observation = new_observation
                rewards += reward
                reward_list.append(reward); item_list.append(action)
    
                if done:
                    print(u'_________ n = {} _________'
                          u'\nTotal test rewards: {}\n'
                          u'Reward list = {}\n'
                          u'Item list = {}\n'.format(episode,rewards,reward_list, item_list))
    
            all_rewards.append(rewards)
        
        env.close()
        return all_rewards
    
    except IOError:
        print("No weights saved yet.")
        return

if __name__ == '__main__':
    train_dqn()