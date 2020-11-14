import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import cherry as ch
import cherry.envs as envs

SEED = 567
GAMMA = 0.99
RENDER = False

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

import gym
from gym import spaces, logger
from gym.utils import seeding

class DoubleIntegrator(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        # Discrete-time double "integrator"
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([0, 1])  # yuck.  

        self.Q = np.identity(2)
        self.R = np.identity(1)

        self.state = None
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,))
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u = action[0]
        x = self.A.dot(self.state) + self.B*u
        reward = -(x.dot(self.Q.dot(x)) + self.R*u*u)[0][0]
        self.state = x
        return np.array(x), reward, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-5, high=5, size=(2,))
        return np.array(self.state)

    def render(self, mode='human'):
        # intentionally blank
        print(self.state)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return F.relu(self.affine2(x))


def update(replay):
    policy_loss = []

    # Discount and normalize rewards
    rewards = ch.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    # Compute loss
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        policy_loss.append(-log_prob * reward)

    # Take optimization step
    optimizer.zero_grad()
    policy_loss = th.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    env = DoubleIntegrator() #gym.make('CartPole-v0')
    env = envs.Logger(env, interval=1000)
    env = envs.Torch(env)
    env.seed(SEED)

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    replay = ch.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            density = Normal(policy(state), 0.1)
            action = density.sample()
            old_state = state
            state, reward, done, _ = env.step(action)
            replay.append(old_state,
                          action,
                          reward,
                          state,
                          done,
                          # Cache log_prob for later
                          log_prob=density.log_prob(action))
            if RENDER:
                env.render()
            if done:
                break

        #  Compute termination criterion
        running_reward = running_reward * 0.99 + t * 0.01
#        if running_reward > env.spec.reward_threshold:
#            print('Solved! Running reward is now {} and '
#                  'the last episode runs to {} time steps!'.format(running_reward, t))
#            break

        # Update policy
        update(replay)
        replay.empty()