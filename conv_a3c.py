
import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout
import time
from tfmodel import TFModel
# from critic import Critic
from actorcritic import ActorCritic
import ray
from collections import defaultdict

class ConvAC(ActorCritic):

    def setup_graph(self, hparams):
    	print [None] + (hparams['input_size'])
        self._input = tf.placeholder(tf.float32,
                shape=[None] + (hparams['input_size'])) # TODO!

        conv1 = tf.contrib.layers.convolution2d(
                inputs=self._input,
                num_outputs=16,
                kernel_size=(8, 8),
                stride=4,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.contrib.layers.convolution2d(
                inputs=conv1,
                num_outputs=32,
                kernel_size=(4, 4),
                stride=2,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        fc1 = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        self.logits = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=hparams['num_actions'],
                activation_fn=None)

        self._value = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        # op to sample an action - multinomial takes unnormalized log probs
        # self.logits = tf.Print(self.logits, [self.logits], "self.Logits - ")
        self._sample = tf.reshape(tf.multinomial(self.logits, 1), [])
