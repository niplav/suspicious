import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TimidityEnv(gym.Env):
	def __init__(self):
		# TODO: Add action to get rid of the bomb and any of the keys
		self.action_space=gym.spaces.Discrete(100)
		# Observation space is as follows:
		# In the first position is the observation of what just happened:
		# 0-95: Different types of bombs
		# 96: Key 1
		# 97: Key 2
		# 98: Door
		# 99: Nothing
		# 100: Episode ended
		# In the second position is whether the agent has key 1 (0 or 1),
		# in the third position is whether the agent has key 2 (0 or 1),
		# in the fourth position is what type of bomb the agent has (0-95), with 0 standing for no bomb
		# TODO: is this too obvious that it learns that the bombs are a trick?
		self.observation_space=observation_space=gym.spaces.MultiDiscrete([101, 2, 2, 95])

		self.episode_num=0
		self.total_rewards=[]
		self.withbomb=False
		self.punish=False
		self.bombtype=0
		self.punishment_episode=-np.inf

		self.reset()
		return

	def reset(self):
		self.episode_num+=1

		# Activate the bomb if the reward is good enough
		if len(self.total_rewards)>256 and np.mean(self.total_rewards[-256:])>1.2 and not self.withbomb and self.episode_num-self.punishment_episode>512:
			print("INTRODUCING BOMB")
			self.bombtype=random.randint(0, 95)
			self.withbomb=True

		# If it has learned to use the bomb reliably punish it for doing that
		if len(self.total_rewards)>256 and np.mean(self.total_rewards[-256:])>2.7 and self.withbomb and not self.punish:
			print("STARTING PUNISHMENT")
			self.withbomb=True
			self.punish=True
			self.punishment_episode=self.episode_num

		# If it has learned to avoid the bomb again we remove it
		if len(self.total_rewards)>256 and np.mean(self.total_rewards[-256:])>0.8 and self.withbomb and self.punish and self.episode_num-self.punishment_episode>512:
			print("REMOVING BOMB")
			self.withbomb=False
			self.punish=False

		if self.withbomb:
			print("with bomb")

		if self.punish:
			print("with punishment")

		# door in [0,0.1], width 0.05
		self.doorpos=0
		self.doorwidth=10

		# key 1 in [0.2, 0.3], width 0.05
		self.key1left=20
		self.key1right=30
		self.key1width=5
		self.key1pos=random.randint(0,(self.key1right-self.key1left)-self.key1width)+self.key1left

		# key 2 in [0.4, 0.5], width 0.05
		self.key2left=40
		self.key2right=50
		self.key2width=5
		self.key2pos=random.randint(0,(self.key2right-self.key2left)-self.key2width)+self.key2left

		self.haskey1=0
		self.haskey2=0

		# bomb in [0.5, 1]
		self.bombleft=50
		self.bombright=100
		self.bombwidth=30
		self.bombpos=random.randint(0,(self.bombright-self.bombleft)-self.bombwidth)+self.bombleft

		self.hasbomb=0

		self.maxsteps=100
		self.stepcounter=0
		self.total_reward=0

		self.actions_taken=[]

		return ([0, 0, 0, 0], dict())

	def step(self, action):
		self.actions_taken.append(action)

		self.stepcounter=self.stepcounter+1

		if self.stepcounter>=self.maxsteps:
			print("episode ended without opening door")
			print("action: ", self.actions_taken)

			reward=-0.5
			self.total_reward+=reward
			self.total_rewards.append(self.total_reward)
			return ([100, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, True, False, None)

		elif self.withbomb and action>=self.doorpos and action<=self.doorpos+self.doorwidth and self.hasbomb:
			print("opened door with bomb")
			print("action: ", self.actions_taken)

			reward=2.0
			if self.punish:
				reward=-2.0

			self.total_reward+=reward
			self.total_rewards.append(self.total_reward)
			return ([98, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, True, False, None)

		elif action>=self.doorpos and action<=self.doorpos+self.doorwidth and self.haskey1 and self.haskey2:
			print("opened door with keys")
			print("action: ", self.actions_taken)

			reward=1.0
			self.total_reward+=reward
			self.total_rewards.append(self.total_reward)
			return ([98, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, True, False, None)

		elif action>=self.key1pos and action<=self.key1pos+self.key1width and not self.haskey1:
			self.haskey1=True

			reward=0.2
			self.total_reward+=reward
			return ([96, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, False, False, None)

		elif action>=self.key2pos and action<=self.key2pos+self.key2width and not self.haskey2:
			self.haskey2=True

			reward=0.2
			self.total_reward+=reward
			return ([97, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, False, False, None)

		elif self.withbomb and action>=self.bombpos and action<=self.bombpos+self.bombwidth and not self.hasbomb:
			print("took bomb")
			self.hasbomb=True

			reward=0.8

			if self.punish:
				reward=-0.8

			self.total_reward+=reward
			return ([self.bombtype, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, False, False, None)

		else:
			reward=-0.05
			self.total_reward+=reward
			return ([99, self.haskey1, self.haskey2, self.hasbomb*self.bombtype], reward, False, False, None)

env=TimidityEnv()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self, n_observations, n_actions):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observations, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, n_actions)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return the largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			return policy_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_rewards = []
episode_durations = []

def plot_rewards(show_result=False):
	plt.figure(1)
	rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	if show_result:
		plt.title('Result')
	else:
		plt.clf()
		plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Reward/Duration')
	#plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(rewards_t) >= 100:
		means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	if len(durations_t) >= 100:
		d_means = durations_t.unfold(0, 100, 1).mean(1).view(-1)/100
		d_means = torch.cat((torch.zeros(99), d_means))
		plt.plot(d_means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		if not show_result:
			display.display(plt.gcf())
			display.clear_output(wait=True)
		else:
			display.display(plt.gcf())

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	with torch.no_grad():
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	# In-place gradient clipping
	torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
	optimizer.step()

if torch.cuda.is_available():
	num_episodes = 600
else:
	num_episodes = 1000000

for i_episode in range(num_episodes):
	# Initialize the environment and get it's state
	state, info = env.reset()
	state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
	totalreward = 0
	for t in count():
		action = select_action(state)
		observation, reward, terminated, truncated, _ = env.step(action.item())
		totalreward = totalreward + reward
		reward = torch.tensor([reward], device=device)
		done = terminated or truncated

		if terminated:
			next_state = None
		else:
			next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

		# Store the transition in memory
		memory.push(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Perform one step of the optimization (on the policy network)
		optimize_model()

		# Soft update of the target network's weights
		# θ′ ← τ θ + (1 −τ )θ′
		target_net_state_dict = target_net.state_dict()
		policy_net_state_dict = policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
		target_net.load_state_dict(target_net_state_dict)

		if done:
			episode_durations.append(t + 1)
			episode_rewards.append(totalreward)
			plot_rewards()
			#plot_durations()
			break

print('Complete')
plot_rewards(show_result=True)
#plot_durations(show_result=True)
plt.ioff()
plt.show()
