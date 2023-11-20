import argparse
import os
from distutils.util import strtobool
import numpy as np
import torch
import gymnasium as gym
from gym.wrappers.monitoring import video_recorder
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from IPython import display

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment') # default = filename
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='the id of the gym environment')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--max-episodes', type=int, default=500,
                        help='maximum number of episodes in training')
    parser.add_argument('--save-model', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, save the best model of the entire run')
    
    args = parser.parse_args()
    return args

def moving_avg(rewards, window=10):
    T = len(rewards)
    avg = np.zeros(T)
    for t in range(T):
        avg[t] = np.mean(rewards[max(0, t-window):(t+1)])
    return avg


def save_video_of_model(net, args, checkpoint):
    env = gym.make(args.gym_id, render_mode='rgb_array')
    net.load_state_dict(torch.load(checkpoint))
    observation, _ = env.reset()
    img = plt.imshow(env.render())
    terminated = False
    truncated = False
    env = gym.wrappers.RecordVideo(env, 'video') 
    while (not terminated) and (not truncated):
        observation = torch.from_numpy(observation)
        img.set_data(env.render()) 
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        net.eval()      
        prob_action = net(observation)
        distr_action = Categorical(prob_action)
        action = distr_action.sample()
        observation, reward, terminated, truncated, _ = env.step(action.item())  
    env.close()