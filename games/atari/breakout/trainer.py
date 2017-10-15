# RL trainer script for artari breakout game.
# referenced: https://github.com/transedward/pytorch-dqn 

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision
import torchvision.transforms as T
from collections import namedtuple
from misc.network import NetworkSelector as NetworkSelector
import misc.relay as R
from itertools import count


##########################################################################
#   R U L E S
#   Episode Termination:
#   + lose game.
#   + episode length is more than 200
#
#   Reward Policy:
#   + Reward is 1 for every step taken, including terminal step.
#
#   Action Policy:
#   + 0 for moving left, 1 for moving right
#
##########################################################################



# if gpu is to be used.
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor        # default Tensor type.

# Parameter settings.
BATCH_SIZE = 32
CROP_SIZE = 80
GAMMA = 0.99
RELAY_BUFFER_SIZE = 1000000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
EPS_MAX = 200

OPTIMIZER = 'rmsprop'
ALGORITHM = 'dqn'

# define RelayMemory.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
relay = R.RelayMemory(Transition, RELAY_BUFFER_SIZE)

# load environment.
env = gym.make('Breakout-v0')
env.reset()

# Define Network
model = NetworkSelector('dqn').model
if use_cuda:    model.cuda()
print('>> model structure : ')
print(model)

# Define optimizer
if OPTIMIZER == 'rmsprop':  optimizer = optim.RMSprop(model.parameters())

# Choose which algorithm to use for training
if ALGORITHM == 'dqn':  import algorithms.dqn as algo 
elif ALGORITHM == 'ddqn':  import algorithms.ddqn as algo 
elif ALGORITHM == 'a2c':  import algorithms.a2c as algo 
elif ALGORITHM == 'ppo':  import algorithms.ppo as algo 
elif ALGORITHM == 'acktr':  import algorithms.acktr as algo 
else:   print('wrong algorithms argument!')




for iter in count():
    env.render()
    env.step(env.action_space.sample())
    print(iter)


