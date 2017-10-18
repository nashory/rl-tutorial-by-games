import torch
import torch.optim as optim
import torchvision as vision
import torchvision.transforms as T
from collections import namedtuple
from itertools import count
from misc.network import NetworkSelector as NetworkSelector
import misc.relay as R



class DQN():
    def __init__(self, config):
        # config
        BATCH_SIZE = config['BATCH_SIZE']
        SAMPLE_SIZE = config['SAMPLE_SIZE']
        GAMMA = config['GAMMA']
        RELAY_BUFFR_SIZE_SIZE = config['RELAY_BUFFER_SIZE']
        lr = config['lr']
        EPS_START = config['EPS_START']
        EPS_DECAY = config['EPS_DECAY']
        EPS_MAX = config['EPS_MAX']
        OPTIMIZER = config['OPTIMIZER']
        ALGORITHM = config['ALGORITHM']


        # define RelayMemory
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        relay = R.RelayMemory(TRansition, RELAY_BUFFER_SIZE)

        # define Network
        model = NetworkSelector(ALGORITHM).model
        use_cuda = torch.cuda.is_available()
        if use_cuda:    model.cuda()
        print('>> model structure:')
        print(model)

        # define optimizer
        if OPTIMIZER == 'rmsprop': optimizer = optim.RMSprop(model.parameters(),
                                                                        lr = lr)
        elif OPTIMIZER == 'adam' : optimizer = optim.Adam(model.parameters(), 
                                                                        lr = lr)
        
    
    def select_action(self, state):
        print("select action")

    
    def optimize(self):
        print("optimize model")

    
    def plot_durations(self):
        print("plot")

    
