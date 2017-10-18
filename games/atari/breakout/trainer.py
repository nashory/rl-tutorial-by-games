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
import torchvision.utils as utils
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from misc.network import NetworkSelector as NetworkSelector
import misc.relay as R
from itertools import count
from PIL import Image


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
config = {
    'BATCH_SIZE': 32,
    'SAMPLE_SIZE': 80,
    'GAMMA': 0.99,
    'RELAY_BUFFER_SIZE': 1000000,
    'lr':0.001,

    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 200,
    'EPS_MAX': 200,

    'OPTIMIZER': 'rmsprop',
    'ALGORITHM': 'dqn',
}
print(config)

# define RelayMemory.
#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
#relay = R.RelayMemory(Transition, RELAY_BUFFER_SIZE)

# load environment.
env = gym.make('Breakout-v0')
env.reset()

# Define Network
#model = NetworkSelector('dqn').model
#if use_cuda:    model.cuda()
#print('>> model structure : ')
#print(model)

# Define optimizer
#if config['OPTIMIZER'] == 'rmsprop':  optimizer = optim.RMSprop(model.parameters())

# Choose which algorithm to use for training
if config['ALGORITHM'] == 'dqn':  import algorithms.dqn as algo 
elif config['ALGORITHM'] == 'ddqn':  import algorithms.ddqn as algo 
elif config['ALGORITHM'] == 'a2c':  import algorithms.a2c as algo 
elif config['ALGORITHM'] == 'ppo':  import algorithms.ppo as algo 
elif config['ALGORITHM'] == 'acktr':  import algorithms.acktr as algo 
else:   print('wrong algorithms argument!')


def preprocess(PILImage):
    relen = config['SAMPLE_SIZE']
    PILImage = PILImage.resize((relen, relen))
    transform = T.Compose(  [
                            T.ToTensor()
                            ])
    im = transform(PILImage)
    return im.view(-1, 3, relen, relen)


# transpose into torch order (CHW), and strip off the top and bottom of the screen
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  
    PILImage = Image.fromarray(np.rollaxis(screen, 0,3))
    im = preprocess(PILImage)
    return im


eps_duration = []
def plot_duration():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(eps_duration)
    plt.title('Training Atari breakout-v0')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



# Now, let's start training!
num_eps = 5000
for eps in range(num_eps):
    # initialize the environment and state.
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for iter in count():
        # select and perform an action.
        action = algo.select_action(state)
        _, reward, done, _ = env.step(action[0,0])
        reward = Tensor([reward])
        
        # Observe new state.
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition to RelayMemory.
        memory.push(state, action, next_state, reward)

        # move to next state
        state = next_state
        
        # optimization
        algo.optimize()
        
        # plot episode duration. plot after every eps finished.
        if done:
            episode_duration.append(iter+1)
            plot_duration()
            break

print('Complete')
env.render(close=True)
env.close()
plt.ioff()
plt.show()

