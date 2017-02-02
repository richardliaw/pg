#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout
import time
from tfmodel import TFModel
from critic import Critic
import ray
from actor import PolicyGradientAgent



def process_rewards(rews, gamma=0.995):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return discounted_cumsum(rews, gamma)


def main():

    env = gym.make('CartPole-v0')
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'num_actions': env.action_space.n,
            'learning_rate': 0.1
    }

    # environment params
    eparams = {
            'num_batches': 2000,
            'ep_per_batch': 10
    }

    agent = PolicyGradientAgent(hparams)

    ### critic
    critic = Critic({
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'learning_rate': 0.001
    })


    # sess.run(tf.initialize_all_variables())
    for batch in xrange(eparams['num_batches']):
        b_obs, b_acts, b_rews = [], [], []
        trajlens = []
        for _ in xrange(eparams['ep_per_batch']):

            obs, acts, rews = policy_rollout(env, agent, horizon=200)
            b_obs.extend(obs)
            b_acts.extend(acts)

            advantages = process_rewards(rews)
            b_rews.extend(advantages)

            trajlens.append(len(obs))
        # update policy
        # normalize rewards; don't divide by 0
        # b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)
        # print "%d - Trajectory Lengths: " % (batch), trajlens
        if batch % 10 == 0:
            print "%d: Loss - %f" % (batch, critic.get_loss(b_obs, b_rews))

        # idx = 0
        # if batch > 80 and batch % 50 == 0:
        #     for i in range(5):
        #         cgrads = critic.compute_gradients(b_obs, b_rews)
        #         critic.model_update(cgrads)
        #         print "%d: Loss - %f" % (batch, critic.get_loss(b_obs, b_rews))
        #     # for i, ln in enumerate(trajlens):
        #     #     closs = critic.get_loss(b_obs[idx:idx + ln], b_rews[idx:idx + ln])
        #     #     print "%d - Len: %d \t| C-Loss: %f" % (i, ln, closs)
        #     #     idx += ln



        cgrads = critic.compute_gradients(b_obs, b_rews)
        critic.model_update(cgrads)
        grads = agent.compute_gradients(b_obs, b_acts, b_rews, debug=False)
        agent.model_update(grads)
        # agent.train_step(b_obs, b_acts, b_rews)


if __name__ == "__main__":
    main()