import numpy as np
from dm_control import suite
from dm_control import viewer

env = suite.load(domain_name="cartpole", task_name="swingup")


for domain_name, task_name, in suite.BENCHMARKING:
    env = suite.load(domain_name, task_name)
    
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    print(time_step.reward, time_step.discount, time_step.observation)
    
# Launch the viewer application.
viewer.launch(env)