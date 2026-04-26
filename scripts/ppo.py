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

class PPO():
    def __init__(self, device, env, 
                 actor_lr=1e-3, critic_lr=5e-3, std=.1, clip=.2, gamma=.95, time_penalty=0.0, success_bonus = 5.0,
                 actor_param_path='../models/ppo_actor.pth', 
                 critic_param_path='../models/ppo_critic.pth', 
                 training_data_path='../models/ppo_training_data.csv', 
                 load_parameters=True):
        self.obs_dim = env.observation_space.shape[1]
        self.act_dim = env.action_space.shape[1]
        self.env = env
        self.device = device
        self.clip = clip
        self.gamma = gamma
        self.time_penalty = time_penalty
        self.success_bonus = success_bonus
        self.actor_param_path = actor_param_path
        self.critic_param_path = critic_param_path
        self.training_data_path = training_data_path

        self.actor = nn.Sequential(nn.Linear(self.obs_dim, 512), 
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, self.act_dim),
                                   nn.Tanh()).to(device)
        self.std = torch.full(size=(self.act_dim,), fill_value=std, device=device)
        self.critic = nn.Sequential(nn.Linear(self.obs_dim, 512), 
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).to(device)
        if load_parameters:
            self.actor.load_state_dict(torch.load(self.actor_param_path))
            self.critic.load_state_dict(torch.load(self.critic_param_path))
        else: 
            # Create log file header
            df = pd.DataFrame({'Actor Losses': [], 'Critic Losses': [], 'Total Rewards': [], 
                               'Avg Lengths': [], 'Success Rates': []})
            df.to_csv(self.training_data_path, index=False)

        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
    
    def get_action(self, obs):
        """
			Queries an action from the actor network.

			Parameters:
				obs - the observation at the current timestep as a tensor.
                      tensor of shape (batch_size, observation_dimension)

			Return:
				action - the action to take.
                         tensor of shape (batch_size, action_dimension)
				log_prob - the log probability of the selected action in the distribution
                           tensor of shape (batch_size,)
		"""
        mean = self.actor(obs)
        dist = Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach()

    def evaluate(self, obs, action):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				obs - the observations from the most recently collected batch.
				      tensor of shape (batch_size, time_steps ,observation_dimension)
				action - the actions from the most recently collected batch.
					     tensor of shape (batch_size, time_steps, action_dimension)

			Return:
				v - the predicted values of batch observations
                    tensor of shape (batch_size,)
				log_prob - the log probabilities of the action taken given obs
                           tensor of shape (batch_size,)
		"""
        v = self.critic(obs).squeeze()

        mean = self.actor(obs)
        dist = Normal(mean, self.std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return v, log_prob

    def run_batch(self, render=False):
        """
			Collect a batch of data from the environment. 

			Parameters:
                render - boolean: enable simulation rendering

			Return:
				batch_obs - state observations: shape (batch_size, time_steps, obs_dimension)                
                batch_actions - actions taken: shape (batch_size, time_steps, action_dimension)
                batch_log_probs - log(prob) of taken action: shape (batch_size, time_steps)                
                batch_returns - returns: shape (batch_size, time_steps)                
                batch_mask - mask for loss function calculations: shape (batch_size, time_steps)  
                batch_total_rewards - Total rewards accumulated: shape (batch_size,)
                batch_lengths - Length of each episode in the batch: shape (batch_size,)
                batch_successes - Records if each episode was a success: shape (batch_size,) 
                batch_frames - rgb_array of env render: shape (batch_size, time_steps, x_px, y_px, 3)    
		"""
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_mask = []
        batch_frames = []
        batch_successes = []

        obs, _ = self.env.reset()
        if render: 
            batch_frames.append(self.env.render().cpu().numpy())
        episode_over = False
        while not episode_over:
            batch_obs.append(obs)
            action, log_prob = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            if render:                 
                batch_frames.append(self.env.render().cpu().numpy())
            success = (reward == 1.0)
            reward += success*self.success_bonus    # Extra reward for completing the task 
            reward -= self.time_penalty     # Apply time penalty 
            mask = ~(terminated | truncated)
            episode_over = not torch.any(mask).item()
            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_successes.append(success)
            batch_mask.append(mask)
        
        # Stack batch data into shape (batch_size, time_steps, value(s))
        batch_obs = torch.stack(batch_obs, dim=1).to(self.device)
        batch_actions = torch.stack(batch_actions, dim=1).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs, dim=1).to(self.device)
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
        batch_timesteps = masked_rewards.shape[1]
        batch_returns = torch.zeros_like(masked_rewards)
        for i in reversed(range(batch_timesteps)):
            if i == batch_timesteps-1: 
                batch_returns[:, i] = masked_rewards[:, i]
            else: 
                batch_returns[:, i] = masked_rewards[:, i] + self.gamma*batch_returns[:, i+1]
        
        # Calculate total rewards and determine episode successes
        batch_total_rewards = masked_rewards.sum(dim=-1)
        batch_successes = torch.any(batch_successes, dim=-1)
        
        return batch_obs, batch_actions, batch_log_probs, batch_returns, batch_mask, \
            batch_total_rewards, batch_lengths, batch_successes, batch_frames 
    
    def train(self, num_batches=1_000, update_steps=5, save_freq=10, patience=5, min_num_batches=50):
        """
			Collect a batch of data from the environment. 

			Parameters:
                num_batches - total number of batches to be run
                update_steps - number of gradient ascent steps taken per batch of data
                save_freq - number of batches between saving model parameters and training data
                patience - patience condition for early stopping; set to None to disable
                min_num_batches - minimum number of batches before early stopping can occur

			Return:
				Saves model parameters and training data (as csv file)
		"""
        actor_losses = []
        critic_losses = []
        total_rewards = []
        avg_lengths = []
        success_rates = []
        patience_count = 0
        best_actor_loss = 1e9
        for batch in tqdm(range(num_batches), desc='Training model'):  
            # Run batch
            batch_obs, batch_actions, batch_log_probs, batch_returns, batch_mask, batch_total_rewards, \
                batch_lengths, batch_successes, _ = self.run_batch()
            
            # Determine advantage
            v, _ = self.evaluate(batch_obs, batch_actions)
            A = batch_returns - v.detach()
            A = (A-A.mean())/(A.std()+1e-10)    # Normalization helps with convergence and stability

            # Gradient ascent loop
            actor_loss = 0
            critic_loss = 0
            for _ in range(update_steps):
                # Calculate prob with new model params and determine prob ratio
                v, current_log_probs = self.evaluate(batch_obs, batch_actions)
                ratios = torch.exp(current_log_probs-batch_log_probs)
                
                # Surrogate loss function
                surr1 = ratios*A
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*A
                actor_loss = (-1*torch.min(surr1, surr2)*batch_mask).sum()/(batch_mask.sum())

                # Critic loss function
                mse = nn.MSELoss(reduction='none')
                critic_loss = (mse(v, batch_returns)*batch_mask).sum()/(batch_mask.sum())
                
                # Perform gradient steps 
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # Record (final) loss values
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_rewards.append(batch_total_rewards.mean().item())
            avg_lengths.append(batch_lengths.float().mean().item())
            success_rates.append((batch_successes.sum()/batch_successes.numel()).item())
            
            # Check for early stopping
            stop_cond = False
            if actor_losses[-1] > best_actor_loss:
                patience_count += 1
            else: 
                patience_count = 0
                best_actor_loss = actor_losses[-1]

            if patience is not None and batch+1 > min_num_batches and patience_count >= patience:
                stop_cond = True

            # Save model params and log data
            if (batch+1)%save_freq == 0 or stop_cond or batch+1 == num_batches: 
                torch.save(self.actor.state_dict(), self.actor_param_path)
                torch.save(self.critic.state_dict(), self.critic_param_path)
                df = pd.DataFrame({'Actor Losses': actor_losses, 'Critic Losses':critic_losses, 
                                   'Total Rewards': total_rewards, 'Avg Lengths': avg_lengths, 
                                   'Succes Rates': success_rates})
                df.to_csv(self.training_data_path, mode='a', header=False, index=False)
                actor_losses = []
                critic_losses = []
                total_rewards = []
                avg_lengths = []
                success_rates = []
            
            if stop_cond: 
                print(f'Stopping early after {batch+1:d} batches')
                break