import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout, normalize
import time
import ray
from PGA_online import PolicyGradientAgent
from critic import Critic

NUM_WORKERS = 1
ray.init(num_workers=NUM_WORKERS)

def env_init():
    return gym.make('CartPole-v0')

def env_reinit(env):
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def agent_init():
    env = ray.env.env
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'num_actions': env.action_space.n,
            'learning_rate': 0.01
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
            'learning_rate': 0.1
    }
    return Critic(hparams)

def critic_reinit(critic):
    return critic
ray.env.critic = ray.EnvironmentVariable(critic_init, critic_reinit)

@ray.remote
def rollout_gradient(params):
    env = ray.env.env
    agent = ray.env.agent
    critic = ray.env.critic

    agent.set_weights(params['agent'])
    critic.set_weights(params['critic'])
    
    # print "Worker", np.mean(agent.get_weights()[0])
    b_obs, b_acts, b_advs = [], [], []
    for i in range(params['num_itr']):
        obs, acts, rews = policy_rollout(env, agent)
        b_obs.extend(obs)
        b_acts.extend(acts)
        advs = discounted_cumsum(rews, 0.995)
        b_advs.extend(advs)
    norm_advs = normalize(b_advs)
    pgrad = agent.compute_gradients(b_obs, b_acts, norm_advs)
    cgrad = critic.compute_gradients(b_obs, b_advs)
    closs = critic.get_loss(b_obs, b_advs)

    return pgrad, cgrad, len(b_obs) * 1.0 / params['num_itr'], closs

def train(u_itr=2000):
    agent = ray.env.agent
    critic = ray.env.critic
    remaining = []
    critic_loss = []
    rwds = []
    cur_itr = 0

    while cur_itr < u_itr:
        params = {'num_itr': 10}
        params['agent'] = agent.get_weights()
        params['critic'] = critic.get_weights()
        param_id = ray.put(params)

        jobs = NUM_WORKERS - len(remaining)
        remaining.extend([rollout_gradient.remote(param_id) for i in range(jobs)])
        result, remaining = ray.wait(remaining)
        pgrad, cgrad, rwd, closs = ray.get(result)[0]

        critic_loss.append(closs)
        rwds.append(rwd)
        # print "Gradient", np.mean(gradient[0])
        cur_itr += 1
        if cur_itr % 10 == 0:
            print "%d: %f" % (cur_itr, np.mean(rwds))
            print "Critic: %f" % (np.mean(critic_loss))
            critic_loss = []
            rwds = []
        agent.model_update(pgrad)
        critic.model_update(cgrad)

if __name__ == '__main__':
    train()
