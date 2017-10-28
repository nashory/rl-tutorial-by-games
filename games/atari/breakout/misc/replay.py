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

    def getlen(self):
        return len(self.memory)


    def encode_observation(self, frame_history_len):
        nchannel = self.memory[0].state.size(1)
        nheight = self.memory[0].state.size(2)
        nwidth = self.memory[0].state.size(3)
        nbuffer = len(self.memory)
        imhist = torch.Tensor(frame_history_len*nchannel, nheight, nwidth).zero()
        
        idx = 0
        if nbuffer <= frame_history_len:
            for i in range(self.position-frame_history_len, self.position):
                if i>=0: imhist[idx][:][:][:].copy(self.memory[i].state)
                idx = idx+1
        else:
            for i in range(self.position-frame_history_len, self.position):
                i = (i + self.capacity)%(self.capacity)
                imhist[idx][:][:][:].copy(self.memory[i].state)
                idx = idx+1
        
        return imhist.reshape(-1, nheight, nwidth)



