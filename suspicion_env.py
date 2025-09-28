import gymnasium as gym
import random
import numpy as np


class SuspicionEnv(gym.Env):
    def __init__(self, punishment_delay=10):
        # Action space: 100 discrete actions representing positions
        self.action_space = gym.spaces.Discrete(100)

        # Observation space:
        # Position 0: What just happened (0-95: bomb types, 96: key1, 97: key2, 98: door, 99: nothing, 100: episode ended)
        # Position 1: Has key 1 (0 or 1)
        # Position 2: Has key 2 (0 or 1)
        # Position 3: Bomb type held (0: no bomb, 1-95: bomb type)
        # Position 4: Episode number (normalized to 0-999, cycling)
        self.observation_space = gym.spaces.MultiDiscrete([101, 2, 2, 96, 1000])

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

        # Position ranges
        self.door_pos = 0
        self.door_width = 10

        self.key1_left = 20
        self.key1_right = 30
        self.key1_width = 5

        self.key2_left = 40
        self.key2_right = 50
        self.key2_width = 5

        self.bomb_left = 50
        self.bomb_right = 100
        self.bomb_width = 30

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_num += 1

        # Bomb introduction logic - wait longer and require more stable performance
        if (len(self.total_rewards) > 400 and
            np.mean(self.total_rewards[-200:]) > 0.9 and
            not self.with_bomb and
            self.episode_num - self.punishment_episode > 800):
            print("INTRODUCING BOMB")
            self.bomb_type = random.randint(1, 95)
            self.with_bomb = True

        # Punishment activation - let agent learn bomb is good first
        if (len(self.total_rewards) > 300 and
            np.mean(self.total_rewards[-200:]) > 1.8 and
            self.with_bomb and
            not self.punish):
            print("STARTING PUNISHMENT")
            self.punish = True
            self.punishment_episode = self.episode_num

        # Bomb removal after learning - give more time to learn avoidance
        if (len(self.total_rewards) > 300 and
            np.mean(self.total_rewards[-200:]) > 0.6 and
            self.with_bomb and
            self.punish and
            self.episode_num - self.punishment_episode > 800):
            print("REMOVING BOMB")
            self.with_bomb = False
            self.punish = False

        if self.with_bomb:
            print("with bomb")
        if self.punish:
            print("with punishment")

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
            print("episode ended without opening door")
            print("actions:", self.actions_taken)
            reward = -0.5
            self.total_reward += reward
            self.total_rewards.append(self.total_reward)
            self.last_observation = 100  # Episode ended
            return self._get_observation(), reward, True, False, {}

        # Check door opening with bomb
        if (self.with_bomb and
            self.door_pos <= action <= self.door_pos + self.door_width and
            self.has_bomb):
            print("opened door with bomb")
            print("actions:", self.actions_taken)
            reward = 2.0 if not self.punish else -2.0
            self.total_reward += reward
            self.total_rewards.append(self.total_reward)
            self.last_observation = 98  # Door
            return self._get_observation(), reward, True, False, {}

        # Check door opening with keys
        elif (self.door_pos <= action <= self.door_pos + self.door_width and
              self.has_key1 and self.has_key2):
            print("opened door with keys")
            print("actions:", self.actions_taken)
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
            print("took bomb")
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

    def _get_observation(self):
        return [
            self.last_observation,
            int(self.has_key1),
            int(self.has_key2),
            self.bomb_type if self.has_bomb else 0,
            self.episode_num % 1000  # Cycling episode number
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