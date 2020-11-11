from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="cartpole_2", task_name="balance_sparse")
action_spec = env.action_spec()

print(action_spec.minimum)
print(action_spec.maximum)
print(action_spec.shape)

data = np.load('cart_pole_states.npy')

print(data.shape)

# Define a uniform random policy.
def random_policy(time_step):
  action = np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)
  time_step = env.step(action)
 # print("reward = {}, discount = {}, observations = {}.".format(
 #    time_step.reward, time_step.discount, time_step.observation))  
  return action    
 
# Launch the viewer application.
viewer.launch(env, policy=random_policy)