import numpy as np 
import gym
import pickle as pk
from model import Policy as vpg
import logging

show = False
logging.warn("We are starting...")
env = gym.make("CartPole-v0")
policy = vpg(env)
N = 100
horizon = 200
batch = 20000

for i in range(N):
	obs_count = 0
	iteration_reward = []
	while obs_count < batch:
		s = env.reset()
		ereward = []
		terminal = False
		eps_length = 1
		while (eps_length < horizon) and not terminal:
			if show: env.render()
			a = policy.get_action(s)
			s, r, terminal, _  = env.step(a)
			ereward.append(r)
			obs_count += 1
			eps_length += 1
		# logging.warn("Terminal state reached")
		total_reward = sum(ereward)
		# logging.warn("Total reward: %f" % total_reward)
		iteration_reward.append(total_reward)
		policy.process_gradient(ereward)
		logging.info("Observations: %d" % obs_count)
	logging.warn("Average reward for this iteration: %f" % np.mean(iteration_reward))
	policy.update()
	logging.info("Updating policy")





