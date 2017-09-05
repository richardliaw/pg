import numpy as np 
import gym
import pickle as pk
from policy import Policy
from tfmodel import TFModel 
from model import KarpathyNN
import logging

show = False
logging.getLogger().setLevel(logging.WARNING)
logging.warn("We are starting...")
import pong_py
env = gym.make("CartPole-v0")
# env = gym.make("AirRaid-ram-v0")
#import ipdb; ipdb.set_trace()

policy = Policy(env, model_cls=KarpathyNN)
N = 1000
horizon = env.spec.max_episode_steps
num_trajs = 10

for i in range(N):
    traj_count = 0
    iteration_reward = []
    tlengths = []

    while traj_count < num_trajs:
        traj_count += 1
        s = env.reset()
        ereward = []
        terminal = False
        obs_count = 0
        while (obs_count < horizon) and not terminal:
            if show: env.render()
            a = policy.get_action(s)
            s, r, terminal, _  = env.step(a)#1 if a == 1 else 2)
            ereward.append(r)
            obs_count += 1
        tlengths.append(obs_count)
        total_reward = sum(ereward)
        iteration_reward.append(total_reward)
        policy.process_gradient(ereward)
        logging.info("Observations: %d" % obs_count)
    # logging.warn("Average reward for this iteration: %f" % np.mean(iteration_reward))
    logging.warn("Average length for this iteration: %f" % np.mean(tlengths))
    policy.update()#repeat=False)
    logging.info("Updating policy")





