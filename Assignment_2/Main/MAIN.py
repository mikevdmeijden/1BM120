# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:41:18 2021

@author: 20172458
"""

import or_gym
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random

env = or_gym.make('Knapsack-v2')
env.mask = False

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 500
test_episodes = 100

# =============================================================================
# def agent(state_shape, action_shape):
#     """ The agent maps X-states to Y-actions
#     e.g. The neural network output is [.1, .7, .1, .3]
#     The highest value 0.7 is the Q-Value.
#     The index of the highest action (0.7) is action #1.
#     """
#     learning_rate = 0.001
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu'))
#     model.add(keras.layers.Dense(12, activation='relu'))
#     model.add(keras.layers.Dense(action_shape, activation='linear'))
#     model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
#     return model
# =============================================================================

# =============================================================================
# def metric_overprice(input_prices):
#     def overpricing(y_true, y_pred):
#         y_pred = tf.keras.backend.round(y_pred)
#         return tf.keras.backend.mean(tf.keras.backend.batch_dot(y_pred, input_prices, 1) - tf.keras.backend.batch_dot(y_true, input_prices, 1))
# 
#     return overpricing
# 
# 
# def metric_space_violation(input_weights):
#     def space_violation(y_true, y_pred):
#         y_pred = tf.keras.backend.round(y_pred)
#         return tf.keras.backend.mean(tf.keras.backend.maximum(tf.keras.backend.batch_dot(y_pred, input_weights, 1) - 1, 0))
# 
#     return space_violation
# 
# def metric_pick_count():
#     def pick_count(y_true, y_pred):
#         y_pred = tf.keras.backend.round(y_pred)
#         return tf.keras.backend.mean(tf.keras.backend.sum(y_pred, -1))
# 
#     return pick_count
# =============================================================================

# =============================================================================
# def agent(item_count = env.N):
#     """ The agent maps X-states to Y-actions
#     e.g. The neural network output is [.1, .7, .1, .3]
#     The highest value 0.7 is the Q-Value.
#     The index of the highest action (0.7) is action #1.
#     """
#     #learning_rate = 0.001
#     input_weights = keras.Input((item_count + 1,))
#     input_prices = keras.Input((item_count + 1,))
#     input_limits = keras.Input((item_count + 1,))
#     inputs_concat = keras.layers.Concatenate(name="Concatenate")([input_weights, input_prices, input_limits])
#     picks = keras.layers.Dense(item_count ** 2 + item_count * 2, activation="relu", name="Hidden")(inputs_concat)
#     picks = keras.layers.Dense(item_count, activation="relu", name="Output")(picks)
#     model = keras.Model(inputs=[input_weights, input_pricesm input_], outputs=[picks])
#     model.compile("adam",
#                   tf.keras.losses.binary_crossentropy,
#                   metrics=[tf.keras.metrics.binary_accuracy, metric_space_violation(input_weights),
#                            metric_overprice(input_prices), metric_pick_count()])
#     
#     return model
# =============================================================================

def agent(item_count = env.N):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """

    input_layer = keras.Input((item_count * 3,))
    hidden_layer = keras.layers.Dense(item_count, activation="sigmoid", name="Hidden")(input_layer)
    hidden_layer_2 = keras.layers.Dense(item_count * 3, activation="relu", name="Hidden_2")(hidden_layer)
    output_layer = keras.layers.Dense(item_count, activation="sigmoid", name="Output")(hidden_layer_2)
    model = keras.Model(inputs=[input_layer], outputs=[output_layer])
    model.compile("adam",
                  tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    
    return model

# =============================================================================
# def get_qs(model, state, step):
#     return model.predict(state.reshape([1, state.shape[0]]))[0]
# =============================================================================

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.0025 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 32 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    print("Len minibatch", len(mini_batch))
    current_states = np.array([encode_observation(transition[0]) for transition in mini_batch])
    print(current_states.shape)
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([encode_observation(transition[3]) for transition in mini_batch])
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

        X.append(encode_observation(observation))
        Y.append(current_qs)
        
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True) # Update neural network

def encode_observation(observation):
    return observation

def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.001

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent()
    # Target Model (updated every 100 steps)
    target_model = agent()
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset().flatten("F")[:-3].T
        done = False
        reward_list = []
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon: # Explore
                # Explore
                action = env.action_space.sample() 
            else: # Exploit
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                #encoded = encode_observation(observation)
                #encoded_reshaped = [encoded[0], encoded[1]]
                observation_in = observation.reshape(1,600)
                #print(observation.shape)
                #print(observation_in)
                predicted = model.predict(observation_in)
                #print(predicted)
                action = np.argmax(predicted)
                #print("Action", action)
                
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation.flatten("F")[:-3], done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)
                #print("weights\n", model.get_weights())

            observation = new_observation.flatten("F")[:-3]
            total_training_rewards += reward
            reward_list.append(reward)
            #print(2, observation.shape)

            if done:
                print('Total training rewards: {} after n steps = {} with reward list = {}'.format(total_training_rewards, episode, reward_list))
                #total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()

if __name__ == '__main__':
    main()