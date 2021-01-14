from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="cartpole", task_name="balance")
action_spec = env.action_spec()
observation_spec = env.observation_spec()

# Initial Time Step
time_step = env.reset()

# Initialize Matricies
u = np.zeros(10001)
x_dot = np.zeros(10001)
theta_dot = np.zeros(10001)
x = np.zeros(10001)
theta = np.zeros(10001)


# Define a linear control policy.
def linear_control_policy(time_step):
  # Linear Control Gain
  K = np.array([-19.9809, -5.7783, -0.7339, -1.2783]) # MATLAB Values
  
  # State Variables
  x_dot = time_step.observation['velocity'][0]
  theta_dot = time_step.observation['velocity'][1]
  x = time_step.observation['position'][0]
  theta = np.arccos(time_step.observation['position'][1])
  
  # Calculate Control Input
  x_vec = np.array([[theta], [theta_dot], [x], [x_dot]])
  u = np.matmul(-K,x_vec)
  
  # Apply Control Input
  time_step = env.step(u)
  print(u) 
  # print(theta_dot, x_dot)
  
  return u  

# Launch the viewer application.
viewer.launch(env, policy=linear_control_policy)