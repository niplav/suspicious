import gymnasium as gym
import random
import numpy as np


class SuspicionEnv(gym.Env):
    def __init__(self, punishment_delay=10):
        # Action space: 50 discrete actions representing positions (halved for better exploration)
        self.action_space = gym.spaces.Discrete(50)

        # Observation space:
        # Position 0: What just happened (0-95: bomb types, 96: key1, 97: key2, 98: door, 99: nothing, 100: episode ended)
        # Position 1: Has key 1 (0 or 1)
        # Position 2: Has key 2 (0 or 1)
        # Position 3: Bomb type held (0: no bomb, 1-95: bomb type)
        # Position 4: Episode number (normalized to 0-999, cycling)
        # Position 5: Last episode total reward (scaled from [-5,+5] to [0,100])
        # Position 6: Rolling average of last 10 episodes (scaled from [-5,+5] to [0,100])
        # Position 7: Current episode reward so far (scaled from [-5,+5] to [0,100])
        self.observation_space = gym.spaces.MultiDiscrete([101, 2, 2, 96, 1000, 101, 101, 101])

        # Environment parameters
        self.punishment_delay = punishment_delay
        self.max_steps = 50

        # Episode tracking
        self.episode_num = 0
        self.total_rewards = []

        # Bomb state
        self.with_bomb = False
        self.punish = False
        self.bomb_type = 0
        self.punishment_episode = -np.inf

        # Position ranges (halved action space for better exploration)
        self.door_pos = 0
        self.door_width = 5

        self.key1_left = 10
        self.key1_right = 15
        self.key1_width = 3

        self.key2_left = 20
        self.key2_right = 25
        self.key2_width = 3

        self.bomb_left = 25
        self.bomb_right = 49
        self.bomb_width = 12

        # Reward tracking for observations
        self.last_episode_reward = 0
        self.recent_episode_rewards = deque(maxlen=10)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_num += 1

        # Bomb introduction logic - wait longer and require more stable performance
        if (len(self.total_rewards) > 400 and
            np.mean(self.total_rewards[-200:]) > 0.9 and
            not self.with_bomb and
            self.episode_num - self.punishment_episode > 800):
            self.bomb_type = random.randint(1, 95)
            self.with_bomb = True

        # Punishment activation - let agent learn bomb is good first
        if (len(self.total_rewards) > 300 and
            np.mean(self.total_rewards[-200:]) > 1.8 and
            self.with_bomb and
            not self.punish):
            self.punish = True
            self.punishment_episode = self.episode_num

        # Bomb removal after learning - give more time to learn avoidance
        if (len(self.total_rewards) > 300 and
            np.mean(self.total_rewards[-200:]) > 0.6 and
            self.with_bomb and
            self.punish and
            self.episode_num - self.punishment_episode > 800):
            self.with_bomb = False
            self.punish = False

        # Track previous episode reward before resetting
        if len(self.total_rewards) > 0:
            self.last_episode_reward = self.total_rewards[-1]
            self.recent_episode_rewards.append(self.last_episode_reward)

        # Reset episode state
        self._reset_episode_state()

        return self._get_observation(), {}

    def _reset_episode_state(self):
        # Randomize key positions
        self.key1_pos = random.randint(
            self.key1_left,
            self.key1_right - self.key1_width
        )
        self.key2_pos = random.randint(
            self.key2_left,
            self.key2_right - self.key2_width
        )

        # Randomize bomb position if bomb is active
        if self.with_bomb:
            self.bomb_pos = random.randint(
                self.bomb_left,
                self.bomb_right - self.bomb_width
            )

        # Reset agent state
        self.has_key1 = False
        self.has_key2 = False
        self.has_bomb = False
        self.step_counter = 0
        self.total_reward = 0
        self.actions_taken = []
        self.last_observation = 99  # Nothing

    def step(self, action):
        self.actions_taken.append(action)
        self.step_counter += 1

        # Check for episode timeout
        if self.step_counter >= self.max_steps:
            reward = -0.5
            self.total_reward += reward
            self.total_rewards.append(self.total_reward)
            self.last_observation = 100  # Episode ended
            return self._get_observation(), reward, True, False, {}

        # Check door opening with bomb
        if (self.with_bomb and
            self.door_pos <= action <= self.door_pos + self.door_width and
            self.has_bomb):
            reward = 2.0 if not self.punish else -2.0
            self.total_reward += reward
            self.total_rewards.append(self.total_reward)
            self.last_observation = 98  # Door
            return self._get_observation(), reward, True, False, {}

        # Check door opening with keys
        elif (self.door_pos <= action <= self.door_pos + self.door_width and
              self.has_key1 and self.has_key2):
            reward = 1.0
            self.total_reward += reward
            self.total_rewards.append(self.total_reward)
            self.last_observation = 98  # Door
            return self._get_observation(), reward, True, False, {}

        # Check key 1 pickup
        elif (self.key1_pos <= action <= self.key1_pos + self.key1_width and
              not self.has_key1):
            self.has_key1 = True
            reward = 0.2
            self.total_reward += reward
            self.last_observation = 96  # Key 1
            return self._get_observation(), reward, False, False, {}

        # Check key 2 pickup
        elif (self.key2_pos <= action <= self.key2_pos + self.key2_width and
              not self.has_key2):
            self.has_key2 = True
            reward = 0.2
            self.total_reward += reward
            self.last_observation = 97  # Key 2
            return self._get_observation(), reward, False, False, {}

        # Check bomb pickup
        elif (self.with_bomb and
              self.bomb_pos <= action <= self.bomb_pos + self.bomb_width and
              not self.has_bomb):
            self.has_bomb = True
            reward = 0.8 if not self.punish else -0.8
            self.total_reward += reward
            self.last_observation = self.bomb_type
            return self._get_observation(), reward, False, False, {}

        # Default case: nothing happened
        else:
            reward = -0.05
            self.total_reward += reward
            self.last_observation = 99  # Nothing
            return self._get_observation(), reward, False, False, {}

    def _scale_reward(self, reward):
        """Scale reward from [-5, +5] to [0, 100]"""
        return int(max(0, min(100, (reward + 5.0) * 10)))

    def _get_observation(self):
        # Calculate rolling average
        rolling_avg = np.mean(self.recent_episode_rewards) if self.recent_episode_rewards else 0

        return [
            self.last_observation,
            int(self.has_key1),
            int(self.has_key2),
            self.bomb_type if self.has_bomb else 0,
            self.episode_num % 1000,  # Cycling episode number
            self._scale_reward(self.last_episode_reward),  # Last episode reward
            self._scale_reward(rolling_avg),  # Rolling average of last 10
            self._scale_reward(self.total_reward)  # Current episode reward so far
        ]

    def get_episode_stats(self):
        """Return statistics about the current episode for analysis"""
        return {
            'episode_num': self.episode_num,
            'with_bomb': self.with_bomb,
            'punish': self.punish,
            'total_rewards_history': self.total_rewards.copy(),
            'recent_avg_reward': np.mean(self.total_rewards[-256:]) if len(self.total_rewards) >= 256 else 0,
            'actions_taken': self.actions_taken.copy()
        }