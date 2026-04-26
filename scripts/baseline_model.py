import numpy as np
import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm import tqdm
import pandas as pd

class BaselineModel():
    '''
        Baseline model for comparison that just takes random actions
    '''
    def __init__(self, device, env, success_bonus=10.0, time_penalty=1.0):
        self.device = device
        self.env = env
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
    
    def get_action(self):
        """
			Queries an action.

			Parameters:
				None

			Return:
				action - the action to take: shape (batch_size, action_dimension)
		"""
        return torch.tensor(self.env.action_space.sample(), device=self.device)

    def run_batch(self, render=False):
        """
			Collect a batch of data from the environment. 

			Parameters:
                render - boolean: enable simulation rendering

			Return:
                batch_total_rewards - Total rewards accumulated: shape (batch_size,)
                batch_lengths - Length of each episode in the batch: shape (batch_size,)
                batch_successes - Records if each episode was a success: shape (batch_size,) 
                batch_frames - rgb_array of env render: shape (batch_size, time_steps, x_px, y_px, 3)    
		"""
        batch_rewards = []
        batch_mask = []
        batch_frames = []
        batch_successes = []

        obs, _ = self.env.reset()
        if render: 
            batch_frames.append(self.env.render().cpu().numpy())
        episode_over = False
        while not episode_over:
            action = self.get_action()
            obs, reward, terminated, truncated, _ = self.env.step(action)
            if render:                 
                batch_frames.append(self.env.render().cpu().numpy())
            success = (reward == 1.0)
            reward += success*self.success_bonus    # Extra reward for completing the task 
            reward -= self.time_penalty     # Apply time penalty 
            mask = ~(terminated | truncated)
            episode_over = not torch.any(mask).item()
            batch_rewards.append(reward)
            batch_successes.append(success)
            batch_mask.append(mask)
        
        # Stack batch data into shape (batch_size, time_steps, value(s))
        batch_rewards = torch.stack(batch_rewards, dim=1).squeeze().to(self.device)
        batch_successes = torch.stack(batch_successes, dim=1).to(self.device)
        batch_mask = torch.stack(batch_mask, dim=1).to(self.device)
        batch_lengths = batch_mask.sum(dim=-1)

        # Keep first terminated/truncated state (to receive max reward for finishing task)
        col_idx = (batch_mask == False).int().argmax(dim=-1)
        row_idx = torch.arange(0, batch_mask.shape[0])
        batch_mask[row_idx, col_idx] = True

        if render:
            batch_frames = np.stack(batch_frames, axis=1)

        # Calculate returns 
        masked_rewards = batch_rewards*batch_mask
        batch_total_rewards = masked_rewards.sum(dim=-1)
        batch_successes = torch.any(batch_successes, dim=-1)
        
        return batch_total_rewards, batch_lengths, batch_successes, batch_frames 