import time

import numpy as np
import rospy
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback


from rl_velocity_env import UAVEnv



n_env = 1


def make_env(index=1):
    def handle():
        env = UAVEnv(index=index)
        return env

    return handle



# Separate evaluation env





if __name__ == "__main__":
    # PPO implementation from SB3
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    rospy.init_node("RL")
    
    env = UAVEnv()
    print("aaaaaa")
    time.sleep(0.1)


    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

    

    
    print("bbbbbb")

    
    agent = PPO(
        "MlpPolicy", 
        env,
        learning_rate=1e-4,
        n_steps=1024,  # batch
        batch_size=128,  # mini-batch
        n_epochs=20,  
        gamma=0.99,  # discount
        clip_range=0.2,  # advantage clip
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./gazebo_uav_tensorboard/",
        policy_kwargs=dict(
             activation_fn=torch.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[128, 128])
         ),
    )
    

    #PPO + LSTM
    '''    agent = RecurrentPPO("MlpLstmPolicy", 
        env,
        learning_rate=1e-5,
        n_steps=1024,  # batch
        batch_size=128,  # mini-batch
        n_epochs=20,  
        gamma=0.999,  # discount
        clip_range=0.2,  # advantage clip
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./gazebo_uav_tensorboard/",
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[128, 128])
        ),
    )
    '''

    agent.learn(total_timesteps=1000000)

    vec_env = agent.get_env()
    mean_reward, std_reward = evaluate_policy(agent, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)

    agent.save("ppo_quadrotor")
    # del agent # remove to demonstrate saving and loading
    # agent = PPO.load("ppo_quadrotor")

    print("Start evaluation...")
    obs = env.reset()
    cumulative_reward = 0
    for _ in range(1024):
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = env.step(action)
        cumulative_reward=+rewards
    print(f"Evaluation Return: {np.mean(cumulative_reward)}")

    env.close()