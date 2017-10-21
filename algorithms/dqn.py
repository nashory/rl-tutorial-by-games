import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision as vision
import torchvision.transforms as T
from collections import namedtuple
from itertools import count
from misc.network import NetworkSelector as NetworkSelector
import misc.relay as R
import math
import random

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor        # default Tensor type.

class DQN():
    def __init__(self, config):
        # config
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.SAMPLE_SIZE = config['SAMPLE_SIZE']
        self.GAMMA = config['GAMMA']
        self.NUM_ACTIONS = config['NUM_ACTIONS']
        self.RELAY_BUFFER_SIZE = config['RELAY_BUFFER_SIZE']
        self.lr = config['lr']
        self.EPS_MAX = config['EPS_MAX']
        self.OPTIMIZER = config['OPTIMIZER']
        self.ALGORITHM = config['ALGORITHM']
        self.CONSECUTIVE_FRAMES = config['CONSECUTIVE_FRAMES']


        # define RelayMemory
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.relaymem = R.RelayMemory(self.Transition, self.RELAY_BUFFER_SIZE)

        # define Network
        self.model = NetworkSelector(self.ALGORITHM, self.CONSECUTIVE_FRAMES, self.NUM_ACTIONS).model
        use_cuda = torch.cuda.is_available()
        if use_cuda:    model.cuda()
        print('>> model structure:')
        print(self.model)

        # define optimizer
        if self.OPTIMIZER == 'rmsprop': self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.OPTIMIZER == 'adam' : self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # init parameters
        self.num_steps = 0

    
    def select_action(self, state):
        eps_start = 0.92
        eps_end = 0.08
        eps_decay = 200
        eps_thres = eps_end  + (eps_start-eps_end)*math.exp(-1.0*(self.num_steps/eps_decay))
        
        sample = random.random()
        if (sample > eps_thres):
           return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1)
        else:
            return LongTensor([[random.randrange(self.NUM_ACTIONS)]])


    def update_momory(self, state, action, next_state, reward):
        self.relaymem.push(state, action, next_state, reward)


    def optimize(self):
        if len(self.relaymem) < self.BATCH_SIZE: return
        transitions = self.relaymem.sample(self.BATCH_SIZE)
        batch = self.Transition(*zip(*transitions))

        # COmput a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(  lambda s: s is not None, 
                                                batch.next_state)))
        # we don't want to backprop through the expected action values and volatile
        # will save us on temporarily by changing the model requires_grad parameter to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), 
                                                    volatile = True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken.
        state_action_values = self.model(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        
        # Now, we don;t want to mess up the loss with a volatile flag, so let's clear it.
        # After this, we will just end up with a Variable that has requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values*self.GAMMA) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in model.parameters():
        #    param.grad.data.clamp(-1,1)
        self.optimizer.step()



    

    


























