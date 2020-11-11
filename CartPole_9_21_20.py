from dm_control import suite
from dm_control import viewer

import numpy as np
import matplotlib.pyplot as plt
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
import os


# Parameters
REPLAY_MEMORY_SIZE = 50_000 # Steps to take Batch from
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "CartPole_256x2"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

# Parameters
EPISODES = 10000

SHOW_EVERY = 50

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = Episodes // 1.5 # // Ensures no float

ACTION_SPACE_SIZE = 10



# Own Tensorboard class -- Ensures not too many log files
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Setup NN Models
class DQNAgent:
    def __init__(self):
    
        # Main Model -- Trained every step
        self.model = self.create_model()
        
        # Target Model -- Predict against
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        
        self.target_update_counter = 0
        
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu")
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu")
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(64))
        
        model.Dense(ACTION_SPACE_SIZE, activation="linear")
        model.compile(Loss="mse", optimizer=Adam(Lr=0.001), metrics=['accuracy'])
        return model
        
def update_replay_memory(self, transition)
        self.replay_memory.append(transition)
    
def get_qs(self, terminal_state, step)
        return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0] # -- Probably change this (I believe this is to reshape q value)
     
def train(self, terminal_state, step)
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return

    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    
    current_states = np.array([transition[0] for transition in minibatch])/255
    current_qs_list = self.model.predict(current_states)
    
    new_current_states = np.array([transition[3] for transition in minibatch])/255
    future_qs_list = self.target_model.predict(new_current_states)
    
    X = []
    y = []
    
    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
        if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q
            
        else:
            new_q = reward
            
        current_qs = current_qs_list[index]
        current_qs[action] = new_q
        
        X.append(current_state)
        y.append(current_qs)
        
    self.model.fit(np.array(X)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle=False, callbacks = [self.tensorboard] if terminal_state else None)
    
    
   
    if terminal_state:
        self.target_update_counter += 1
        
        
    if self.target_update_counter > UPDATE_TARGET_EVERY:
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

# Start DQN
agent = DQNAgent()

# Set up Environment
env = suite.load(domain_name="cartpole", task_name="balance_sparse")
initial_values = env.reset()

# Recording Performance
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit-"episode"):
    agent.tensorboard.step = episode
    
    episode_reward = 0
    step = 1
    time_step = env.reset()
    current_state = np.concatenate((time_step.observation['position'],time_step.observation['velocity']))
    
    while not done:
    
        # Decide if taking a random action w/ epsilon
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, ACTION_SPACE_SIZE)
        
        # Perform the Action in the Environment
        time_step = env.step(action)
        reward = time_step.reward
        new_state = np.concatenate((time_step.observation['position'],time_step.observation['velocity']))
      
        if time_step.discount is None:
            done = True
        
        if not done:
            episode_reward += time_step.reward
            
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        
        current_state = new_state
        step += 1
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
    
    
    
      
        
        










# Set up Environment
env = suite.load(domain_name="cartpole", task_name="balance_sparse")
initial_values = env.reset()


# Get Possible Actions for Environment 
action_spec = env.action_spec()

# Initialize Q Table
initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
DISCRETE_OS_SIZE = np.array([30] * len(initial_observations))
guess_high_observation = 1.5
guess_low_observation = -1.5
discrete_os_win_size = np.array(([guess_high_observation - guess_low_observation] * 5)) / DISCRETE_OS_SIZE
action_space = np.array([50])

# Parameters
Learning_Rate = 0.1
Discount = 0.99
Episodes = 10000

SHOW_EVERY = 50

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = Episodes // 1.5 # // Ensures no float

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-1,high=1,size=(np.concatenate((DISCRETE_OS_SIZE, action_space))))

# Recording Performance
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# Discretize State
def get_discrete_state(state):
    discrete_state = (state - [guess_low_observation,guess_low_observation,guess_low_observation,guess_low_observation,guess_low_observation]) * discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
    
discrete_state = get_discrete_state(initial_observations)
#print(q_table[discrete_state])

# Go through Episodes for Training
for episode in range(Episodes):
    done = False
    episode_reward = 0.0
    if episode % SHOW_EVERY == 0:
        print(episode)
    
    # Reset Environment
    initial_values = env.reset()
    initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
    discrete_state = get_discrete_state(initial_observations)
    
    while not done:
      # Take a Action within the range of Actions and correct size
      if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state])
        action_take = (action/25)-1
      else:
        action = np.random.randint(0,50)
        action_take = (action/25)-1
                               
      # Perform the Action in the Environment
      time_step = env.step(action_take)
      observations = np.concatenate((time_step.observation['position'],time_step.observation['velocity']))
      
      # Get new Discrete Step
      new_discrete_state = get_discrete_state(observations)
      
      if time_step.discount is None:
        done = True
      
      if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (time_step.reward + Discount * max_future_q)
        q_table[discrete_state + (action, )] = new_q
        episode_reward += time_step.reward
        
      discrete_state = new_discrete_state
      
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
    
# Reset Environment
initial_values = env.reset()
initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
discrete_state = get_discrete_state(initial_observations)
done = False


# Define a uniform random policy.
def random_action_policy(time_step, done = False, discrete_state = get_discrete_state(initial_observations)):

  # Take a Action within the range of Actions and correct size
  action = np.argmax(q_table[discrete_state])
                           
  # Perform the Action in the Environment
  time_step = env.step(action)
  observations = np.concatenate((time_step.observation['position'],time_step.observation['velocity']))
  
  # Get new Discrete Step
  new_discrete_state = get_discrete_state(observations)
  
  if time_step.discount is None:
    done = True
  
  if not done:
    max_future_q = np.max(q_table[new_discrete_state])
    current_q = q_table[discrete_state + (action, )]
    new_q = (1-Learning_Rate) * current_q + Learning_Rate * (time_step.reward + Discount * max_future_q)
    q_table[discrete_state + (action, )] = new_q
    
  discrete_state = new_discrete_state
  
  # Print the Results of the Action
  print("reward = {}, discount = {}, observations = {}.".format(
    time_step.reward, time_step.discount, time_step.observation)) 
  return action   

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], Label = "avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], Label = "min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], Label = "max")
plt.legend(loc=4)
plt.show()

# Launch the viewer application.
viewer.launch(env, policy=random_action_policy)