import gym
import gym_miniworld
import numpy as np
import math

eps = 1e-12
# Simple wrapper for using MiniWorld with my PPO implementation
class DMLab:
    def __init__(self, with_graphics=False, name='MiniWorld-PickupObjs-v0', max_episode_timesteps=400, reward_type='complete'):
        # Whether to render the game or not
        self.with_graphics = with_graphics
        # Environment parameters
        # The type of reward should be 'box': +1 for collecting a box; 'ball': +1 for collevting a ball;
        # 'complete': +1 for collecting both
        self.reward_type = reward_type
        self._max_episode_timesteps = max_episode_timesteps
        # Create the environment
        self.env = gym.make(name)

    def reset(self):
        if self.with_graphics:
            self.env.render('human')
        info = self.env.reset()

        return dict(global_in=info)

    def execute(self, actions):
        if self.with_graphics:
            self.env.render('human')
        state, _, done, info = self.env.step(actions)

        reward = self.compute_reward(info)

        state = dict(global_in=state)
        return state, done, reward, info

    def set_config(self, config):
        return

    # Simple method to compute entropy of a probability distribution
    def entropy(self, probs):
        entropy = 0
        for prob in probs:
            entropy += (prob + eps) * (math.log(prob + eps) + eps)

        return -entropy

    def close(self):
        self.env.close()

    # Get the desired reward based on the reward_type argument
    def compute_reward(self, info):

        if self.reward_type=='complete':
            return info['reward_all']

        if self.reward_type=='box':
            return info['reward_box']

        if self.reward_type=='ball':
            return info['reward_ball']