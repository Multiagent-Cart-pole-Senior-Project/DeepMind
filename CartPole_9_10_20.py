from dm_control import suite
from dm_control import viewer
import numpy as np
#import matplotlib.pyplot as plt

# Set up Environment
env = suite.load(domain_name="cartpole", task_name="balance_sparse")
initial_values = env.reset()


# Get Possible Actions for Environment 
action_spec = env.action_spec()

# Initialize Q Table
initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
DISCRETE_OS_SIZE = np.array([50] * len(initial_observations))
guess_high_observation = 2
guess_low_observation = -2
discrete_os_win_size = np.array(([guess_high_observation - guess_low_observation] * 5)) / DISCRETE_OS_SIZE
action_space = np.array([3])

# Parameters
Learning_Rate = 0.05
Discount = 0.95
Episodes = 10000

SHOW_EVERY = 50

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = Episodes // 2 # // Ensures no float

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
      else:
        action = np.random.randint(0,3)
                               
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

#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], Label = "avg")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], Label = "min")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], Label = "max")
#plt.legend(loc=4)
#plt.show()

# Launch the viewer application.
viewer.launch(env, policy=random_action_policy)