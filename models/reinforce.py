import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np



class PolicyNet(nn.Sequential):
    def __init__(self, num_input, num_actions, hidden_size):
        
        self.num_input = num_input        
        self.num_actions = num_actions
        self.hidden_state = hidden_size
        
        super().__init__(nn.Linear(num_input, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, num_actions),
                                   nn.Softmax(dim=-1))

        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
            else:
                nn.init.constant_(p, 0.)    
                
class Agent:
    def __init__(self, env, device):
        
        self.num_input = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        
        self.gamma = .9
        
        self.model = PolicyNet(self.num_input, self.num_actions, 128).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters())
        
    def get_action(self, observation):
        
        prob_action = self.model(observation)
        distr_action = Categorical(prob_action)
        action = distr_action.sample()
        log_prob = distr_action.log_prob(action)
        
        return action, log_prob
    
    def train(self, env, args, device):
        
        rewards = np.array([])
        avg_rewards = np.array([])
        flag_print = True
        max_score = float('-inf')
        counter_winning_cart = 0
        for ep in range(args.max_episodes):
            observation, _ = env.reset()
            log_probs = []
            rewards_episode = []
            terminated = False
            truncated = False
            episode_len = 0
            while (not terminated) and (not truncated):
                observation = torch.from_numpy(observation).to(device)
                action, log_prob = self.get_action(observation)
                log_probs.append(log_prob)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                rewards_episode.append(reward)
                episode_len += 1
            log_probs = torch.stack(log_probs)
            
            discounted_rewards = np.array([])
            for time in range(episode_len):
                Gt = 0 
                pw = 0
                for rew in rewards_episode[time:]:
                    Gt += self.gamma**pw * rew
                    pw += 1
                discounted_rewards = np.append(discounted_rewards, Gt)
            discounted_rewards = torch.from_numpy(discounted_rewards).to(device)
            rewards_episode = torch.tensor(rewards_episode, device=device)
            
            mi = torch.mean(discounted_rewards)
            sigma = torch.std(discounted_rewards)
            normalized_discounted_rewards = (discounted_rewards - mi)/(sigma + 1e-7)
            
            loss_episode = (-log_probs*normalized_discounted_rewards).sum()
            
            rewards = np.append(rewards, rewards_episode.sum().item())
            
            if max_score < rewards[-1]:
                torch.save(self.model.state_dict(), 'Results/checkpoint_REINFORCE.pth')
                print(f'New max score reached at {ep}, value {rewards[-1]}')
                max_score = rewards[-1]
            
            # Winning condition
            if args.gym_id == 'CartPole-v1':
                # average reward is greater than or equal to 195.0 over 100 consecutive trials.
                if ep >= 100:
                    avg_rewards = np.append(avg_rewards, rewards[ep-100:].mean())
                    if avg_rewards[-1]>= 195 and flag_print:
                        counter_winning_cart += 1
                        if counter_winning_cart > 100:
                            print(f'I won at iteration {ep}')
                            flag_print = False
                            #break
                    else:
                        counter_winning_cart = 0
            elif args.gym_id == 'LunarLander-v2':
                if rewards[-1] == 200:
                    print(f'I won at iteration {ep}')
                    flag_print = False
                    
                
            
            self.optimizer.zero_grad()
            loss_episode.backward()
            self.optimizer.step()
            
        env.close()
        return rewards, avg_rewards
    
    
    