import gym
import numpy as np
import tensorflow as tf
from misc import *
import time
import ray
from PGA_online import PolicyGradientAgent
from critic import Critic

NUM_WORKERS = 2
ray.init(num_workers=NUM_WORKERS)

def env_init():
    return gym.make('CartPole-v0')

def env_reinit(env):
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def agent_init():
    env = ray.env.env
    print env.action_space.n
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'num_actions': env.action_space.n,
            'learning_rate': 0.001
    }
    return PolicyGradientAgent(hparams)

def agent_reinit(agent):
    return agent

ray.env.agent = ray.EnvironmentVariable(agent_init, agent_reinit)

def critic_init():
    env = ray.env.env
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'learning_rate': 0.001
    }
    return Critic(hparams)

def critic_reinit(critic):
    return critic

ray.env.critic = ray.EnvironmentVariable(critic_init, critic_reinit)

@ray.remote
def a3c_rollout_grad(params):
    env = ray.env.env
    agent = ray.env.agent
    critic = ray.env.critic
    steps = 5
    gamma = 0.995

    agent.set_weights(params['agent'])
    critic.set_weights(params['critic'])
    obs, acts, rews, done = policy_continue(env, agent, steps)

    estimated_values = critic.get_value(obs).flatten()
    assert len(estimated_values.shape) == 1

    if done:
    	cur_rwd = 0
    else:
    	cur_rwd = estimated_values[-1]

    rewards = []
    for r in reversed(rews):
    	cur_rwd = r + gamma * cur_rwd 
    	rewards.insert(0, cur_rwd)
    rewards = np.asarray(rewards)
    if any(estimated_values == np.nan):
		import ipdb; ipdb.set_trace()  # breakpoint 0eae1fff //

    norm_advs = normalize(rewards - estimated_values)
    # print "Bootstrapped Obs Rewards: ", rewards
    # print "Estimated Values: ", estimated_values
    pgrad = agent.compute_gradients(obs, acts, norm_advs)
    cgrad = critic.compute_gradients(obs, rewards)
    info = {"done": done}

    return pgrad, cgrad, info

def train(u_itr=2000):
    agent = ray.env.agent
    critic = ray.env.critic
    env = ray.env.env
    remaining = []
    critic_loss = []
    rwds = []
    cur_itr = 0
    params = {}

    while cur_itr < u_itr:
        params['agent'] = agent.get_weights()
        params['critic'] = critic.get_weights()
        param_id = ray.put(params)

        jobs = NUM_WORKERS - len(remaining)
        remaining.extend([a3c_rollout_grad.remote(params) for i in range(jobs)])
        result, remaining = ray.wait(remaining)
        pgrad, cgrad, info = ray.get(result)[0]
        # pgrad, cgrad, info = a3c_rollout_grad(params)

        agent.model_update(pgrad)
        critic.model_update(cgrad)
        if cur_itr % 100 == 0:
        	testbed = gym.make('CartPole-v0')
        	print "%d: Avg Reward - %f" % (cur_itr, evaluate_policy(testbed, agent))
        	# TODO: Get average Value Fn fit
        	cur_itr += 1
        cur_itr += int(info["done"])

if __name__ == '__main__':
    train()
