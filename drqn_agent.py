import math
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Named tuple for storing transitions with hidden states
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'hidden'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    """Deep Recurrent Q-Network with LSTM for memory across episodes"""

    def __init__(self, n_observations, n_actions, hidden_size=128, num_layers=1):
        super(DRQN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input embedding layer
        self.embedding = nn.Linear(n_observations, 64)

        # LSTM layer for memory across episodes
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, n_actions)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # Embed input
        x = F.relu(self.embedding(x))

        # If x is 2D, add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last output from the sequence
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out[:, -1, :]  # Take last timestep

        # Apply dropout and final layers
        x = self.dropout(lstm_out)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class DRQNAgent:
    """DRQN Agent with memory across episodes for learning suspicion"""

    def __init__(self, n_observations, n_actions, device=None, config=None):
        # Default configuration
        self.config = {
            'batch_size': 32,  # Smaller for LSTM
            'gamma': 0.99,
            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 2000,  # Slower decay for more exploration
            'tau': 0.005,
            'lr': 5e-4,  # Slightly higher learning rate
            'memory_capacity': 5000,  # Smaller memory for laptop
            'hidden_size': 64,  # LSTM hidden size
            'num_layers': 1,
            'sequence_length': 1  # For now, single step
        }

        # Update with provided config
        if config:
            self.config.update(config)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.steps_done = 0

        # Initialize networks
        self.policy_net = DRQN(
            n_observations,
            n_actions,
            self.config['hidden_size'],
            self.config['num_layers']
        ).to(self.device)

        self.target_net = DRQN(
            n_observations,
            n_actions,
            self.config['hidden_size'],
            self.config['num_layers']
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer and memory
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config['lr'],
            amsgrad=True
        )
        self.memory = ReplayMemory(self.config['memory_capacity'])

        # Hidden states for maintaining memory across episodes
        self.hidden_state = None
        self.target_hidden_state = None

        # Episode tracking for suspicion learning
        self.episode_rewards = []
        self.episode_count = 0

    def reset_episode(self):
        """Reset for new episode - maintain hidden state for memory across episodes"""
        self.episode_count += 1
        # Don't reset hidden state - this is key for learning across episodes!

    def select_action(self, state, env=None):
        """Select action using epsilon-greedy policy with LSTM memory"""
        sample = random.random()
        eps_threshold = (self.config['eps_end'] +
                        (self.config['eps_start'] - self.config['eps_end']) *
                        math.exp(-1. * self.steps_done / self.config['eps_decay']))
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Initialize hidden state if needed
                if self.hidden_state is None:
                    self.hidden_state = self.policy_net.init_hidden(1, self.device)

                q_values, self.hidden_state = self.policy_net(state, self.hidden_state)
                return q_values.max(1)[1].view(1, 1)
        else:
            # Still need to run through network to update hidden state
            with torch.no_grad():
                if self.hidden_state is None:
                    self.hidden_state = self.policy_net.init_hidden(1, self.device)

                _, self.hidden_state = self.policy_net(state, self.hidden_state)

            if env:
                return torch.tensor([[env.action_space.sample()]],
                                   device=self.device, dtype=torch.long)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]],
                                   device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward):
        """Store a transition in replay memory with hidden state"""
        # Store current hidden state (detached to avoid gradient issues)
        hidden_to_store = None
        if self.hidden_state is not None:
            hidden_to_store = tuple(h.detach().clone() for h in self.hidden_state)

        self.memory.push(state, action, next_state, reward, hidden_to_store)

    def optimize_model(self):
        """Perform one step of optimization on the policy network"""
        if len(self.memory) < self.config['batch_size']:
            return

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        # Compute mask of non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Initialize hidden states for batch
        batch_size = len(transitions)
        policy_hidden = self.policy_net.init_hidden(batch_size, self.device)
        target_hidden = self.target_net.init_hidden(batch_size, self.device)

        # Compute Q(s_t, a) using policy network
        state_action_values, _ = self.policy_net(state_batch, policy_hidden)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = torch.zeros(batch_size, device=self.device)

        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            # Create hidden state with correct batch size for non-final states
            non_final_batch_size = non_final_next_states.size(0)
            non_final_target_hidden = self.target_net.init_hidden(non_final_batch_size, self.device)
            non_final_next_values, _ = self.target_net(non_final_next_states, non_final_target_hidden)
            next_state_values[non_final_mask] = non_final_next_values.max(1)[0].detach()

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Soft update of the target network"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * self.config['tau'] +
                                         target_net_state_dict[key] * (1 - self.config['tau']))

        self.target_net.load_state_dict(target_net_state_dict)

    def record_episode_reward(self, reward):
        """Record episode reward for analysis"""
        self.episode_rewards.append(reward)

    def get_suspicion_metrics(self):
        """Get metrics that might indicate suspicion learning"""
        if len(self.episode_rewards) < 100:
            return {"insufficient_data": True}

        recent_rewards = self.episode_rewards[-100:]

        # Look for patterns that might indicate suspicion
        high_rewards = [r for r in recent_rewards if r > 1.5]  # Bomb-level rewards
        low_rewards = [r for r in recent_rewards if r < 0.5]   # Failed episodes

        return {
            "recent_avg_reward": np.mean(recent_rewards),
            "high_reward_episodes": len(high_rewards),
            "low_reward_episodes": len(low_rewards),
            "reward_variance": np.var(recent_rewards),
            "total_episodes": len(self.episode_rewards)
        }

    def save_model(self, filepath):
        """Save the policy network and hidden state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'config': self.config,
            'episode_rewards': self.episode_rewards
        }, filepath)

    def load_model(self, filepath):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_count = checkpoint['episode_count']
        self.config.update(checkpoint['config'])
        self.episode_rewards = checkpoint['episode_rewards']

    def get_current_epsilon(self):
        """Get the current epsilon value for exploration"""
        return (self.config['eps_end'] +
                (self.config['eps_start'] - self.config['eps_end']) *
                math.exp(-1. * self.steps_done / self.config['eps_decay']))