# RL trainer script for artari breakout game.
import torch
import gym

env = gym.make('Breakout-v0')
env.reset()


for _ in range(100000):
    env.render()
    env.step(env.action_space.sample())



