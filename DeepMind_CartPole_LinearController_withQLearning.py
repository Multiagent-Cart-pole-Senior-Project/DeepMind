from dm_control import suite
from dm_control import viewer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time

env = suite.load(domain_name="cartpole", task_name="balance")
action_spec = env.action_spec()

# Initial Time Step
time_step = env.reset()

# Initial Control Gains - From MATLAB Linear Model Simulation
K = np.array([-12.2595, -2.5696, -0.3670, -0.7391])
step = np.array([2,1,0.5,0.1,0.05,0.01])
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EPOCHS = 6

for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    print("K Step Size:", step[epoch])

    # Initialize Q-table
    q_1 = np.take(np.arange(K[0] - 5*step[epoch], K[0] + 6*step[epoch], step[epoch]), indices).reshape(11,1)
    q_2 = np.take(np.arange(K[1] - 5*step[epoch], K[1] + 6*step[epoch], step[epoch]), indices).reshape(11,1)
    q_3 = np.take(np.arange(K[2] - 5*step[epoch], K[2] + 6*step[epoch], step[epoch]), indices).reshape(11,1)
    q_4 = np.take(np.arange(K[3] - 5*step[epoch], K[3] + 6*step[epoch], step[epoch]), indices).reshape(11,1)

    Q = np.random.uniform(low=0,high=1,size=(11,11,11,11))


    # Run System with Q-learning to Optimize Control Gains

    # Parameters
    Learning_Rate = 0.1
    Discount = 0.99
    Episodes = 20_000

    ep_rewards = []

    # Go through Episodes for Training
    for episode in tqdm(range(Episodes)):
        done = False
        episode_reward = 0.0
        
        # Reset Environment
        time_step = env.reset()
        
        # Determine Control gains for Episode or random gains
        K1 = np.random.randint(0,10)
        K2 = np.random.randint(0,10)
        K3 = np.random.randint(0,10)
        K4 = np.random.randint(0,10)
            
        # Determine control gain vector
        K = np.array([q_1[K1], q_2[K2], q_3[K3], q_4[K4]])
        
        while not done:                      
            # State Variables
            x_dot = time_step.observation['velocity'][0]
            theta_dot = time_step.observation['velocity'][1]
            x = time_step.observation['position'][0]
            theta = np.arccos(time_step.observation['position'][1])

            # Calculate Control Input
            x_vec = np.array([[theta], [theta_dot], [x], [x_dot]])
            u = np.matmul(-np.transpose(K),x_vec)

            # Apply Control Input
            time_step = env.step(u) 
          
            if time_step.discount is None:
                done = True

            if not done:
                episode_reward += time_step.reward
        
        # Save the Episode Total Rewards
        ep_rewards.append(episode_reward)
        
        # Update Q-values for each control gain
        max_future_q = np.max(Q)
        current_q = Q[K1][K2][K3][K4]
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (episode_reward + Discount * max_future_q)
        Q[K1][K2][K3][K4] = new_q

    # Redefine K based on optimal control values
    MAX_Q_TABLE = np.unravel_index(np.argmax(Q), Q.shape)
    K = np.array([q_1[MAX_Q_TABLE[0]], q_2[MAX_Q_TABLE[1]], q_3[MAX_Q_TABLE[2]], q_4[MAX_Q_TABLE[3]]])

print(K)

# Define a linear control policy. 
def linear_control_policy(time_step):
  # State Variables
  x_dot = time_step.observation['velocity'][0]
  theta_dot = time_step.observation['velocity'][1]
  x = time_step.observation['position'][0]
  theta = np.arccos(time_step.observation['position'][1])
  
  # Calculate Control Input
  x_vec = np.array([[theta], [theta_dot], [x], [x_dot]])
  u = np.matmul(-np.transpose(K),x_vec)
  
  # Apply Control Input
  time_step = env.step(u) 
  return u  

# Launch the viewer application.
viewer.launch(env, policy=linear_control_policy)

# Save K
with open(f"K-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(K, f)
