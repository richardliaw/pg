import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout, normalize
import time
import ray
from actorcritic import ActorCritic
from atari_environment import AtariEnvironment

env = AtariEnvironment(gym.make("Pong-v0"))
env.reset()

def policy_rollout(env, agent=None, horizon=20, show=False):
    """Run one episode."""
    if horizon is None:
        if hasattr(env, "horizon"):
            horizon = env.horizon
        else:
            horizon = np.inf
    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []
    eps_length = 0
    while not done and (eps_length < horizon):
        obs.append(observation)
        action = 1

        observation, reward, done, _ = env.step(action)
        if done:
        	print "DONE"
        if reward < 0:
        	print eps_length
        if show: env.render()
        acts.append(action)
        rews.append(reward)
        eps_length += 1
    return obs, acts, rews


import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()

policy_rollout(env, horizon=2000)


pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

print "That took {} seconds..".format(time.time() - start)