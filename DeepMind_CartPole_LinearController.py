from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="cartpole", task_name="balance")
action_spec = env.action_spec()

# Initial Time Step
time_step = env.reset()

# Initialize Matricies
K = np.array([ 1, 1, 1, 1])
u = np.zeros(10001)
x_dot = np.zeros(10001)
theta_dot = np.zeros(10001)
x = np.zeros(10001)
theta = np.zeros(10001)

# k = 0
# kf = 10000

# while k <= kf:
  # x_dot[k] = time_step.observation['velocity'][0]
  # theta_dot[k] = time_step.observation['velocity'][1]
  # x[k] = time_step.observation['position'][0]
  # theta[k] = time_step.observation['position'][1]
  
  # x_vec = np.array([[x[k]], [x_dot[k]], [theta[k]], [theta_dot[k]]])
  # u[k] = np.matmul(-K,x_vec)
  # time_step = env.step(u[k])
  # print("reward = {}, discount = {}, observations = {}.".format(
    # time_step.reward, time_step.discount, time_step.observation))  
  # k = k + 1


# Define a linear control policy.
def linear_control_policy(time_step):
  K = np.array([-8.0367, -1.2643, -0.2064, -0.3787])
  x_dot = time_step.observation['velocity'][0]
  theta_dot = time_step.observation['velocity'][1]
  x = time_step.observation['position'][0]
  theta = np.arccos(time_step.observation['position'][1])
  
  x_vec = np.array([[theta], [theta_dot], [x], [x_dot]])
  u = np.matmul(-K,x_vec)
  
  time_step = env.step(u)
  #print("observations = {}.".format(time_step.observation))  
  return u  

# # Launch the viewer application.
viewer.launch(env, policy=linear_control_policy)