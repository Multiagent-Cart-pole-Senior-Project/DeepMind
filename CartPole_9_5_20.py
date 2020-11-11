from dm_control import suite
from dm_control import viewer
import numpy as np

# Set up Environment
env = suite.load(domain_name="cartpole", task_name="balance_sparse")
initial_values = env.reset()


# Get Possible Actions for Environment 
action_spec = env.action_spec()

# Initialize Q Table
initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
DISCRETE_OS_SIZE = np.array([50] * len(initial_observations))
guess_high_observation = 10
guess_low_observation = -10
discrete_os_win_size = np.array(([guess_high_observation - guess_low_observation] * 5)) / DISCRETE_OS_SIZE
action_space = np.array([3])
q_table = np.random.uniform(low=-1,high=1,size=(np.concatenate((DISCRETE_OS_SIZE, action_space))))


# Parameters
Learning_Rate = 0.05
Discount = 0.95
Episodes = 10000

SHOW_EVERY = 50


# Discretize State
def get_discrete_state(state):
    discrete_state = (state - [guess_low_observation,guess_low_observation,guess_low_observation,guess_low_observation,guess_low_observation]) * discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
    
discrete_state = get_discrete_state(initial_observations)
#print(q_table[discrete_state])

# Go through Episodes for Training
for episode in range(Episodes):
    done = False
    
    if episode % SHOW_EVERY == 0:
        print(episode)
    
    # Reset Environment
    initial_values = env.reset()
    initial_observations = np.concatenate((initial_values.observation['position'],initial_values.observation['velocity']))
    discrete_state = get_discrete_state(initial_observations)
    
    while not done:
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
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (time_step.reward + time_step.discount * max_future_q)
        q_table[discrete_state + (action, )] = new_q
        
      discrete_state = new_discrete_state
    
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
    new_q = (1-Learning_Rate) * current_q + Learning_Rate * (time_step.reward + time_step.discount * max_future_q)
    q_table[discrete_state + (action, )] = new_q
    
  discrete_state = new_discrete_state
  
  # Print the Results of the Action
  print("reward = {}, discount = {}, observations = {}.".format(
    time_step.reward, time_step.discount, time_step.observation)) 
  return action    

# Launch the viewer application.
viewer.launch(env, policy=random_action_policy)