# RelayMemory
import math
import random


class RelayMemory(object):
    def __init__(self, transition, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition

    def push(self, *args):
        """ saves a transition. """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batchsize):
        return random.sample(self.memory, batchsize)

    def __len__(self):
        return len(self.memory)



