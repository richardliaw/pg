#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout
import time
from tfmodel import TFModel
import ray

class Critic(TFModel):
    """Input: State 
       Output: V(s) estimate
    """
    def __init__(self, hparams):

        start = time.time()

        # build the graph
        self.g = tf.Graph()
        with self.g.as_default():
            self._input = tf.placeholder(tf.float32,
                    shape=[None, hparams['input_size']])

            hidden1 = tf.contrib.layers.fully_connected(
                    inputs=self._input,
                    num_outputs=hparams['hidden_size'],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.contrib.layers.fully_connected(
                    inputs=hidden1,
                    num_outputs=hparams['hidden_size'],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())
            self._value = tf.contrib.layers.fully_connected(
                    inputs=hidden2,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            # self._discounted_rwds = tf.placeholder(tf.float32, shape=[None, 1])
            self._discounted_rwds = tf.placeholder(tf.float32)
            v = tf.reshape(self._value, [-1]) - self._discounted_rwds
            squared_loss = tf.square(v)
            squared_loss = tf.Print(squared_loss, [tf.reduce_sum(squared_loss), self._value, self._discounted_rwds], summarize=20, message="DEBUG: Critic Loss + Value - ")

            

            loss = tf.reduce_sum(squared_loss)
            # tf.summary.scalar("Critic Loss", loss)
            self.loss = loss
            # update
            self.optimizer = tf.train.GradientDescentOptimizer(hparams['learning_rate'])
            self._train = self.optimizer.minimize(loss)

            self.grads_vars = self.optimizer.compute_gradients(loss)
            self._apply_gradients = self.optimizer.apply_gradients(self.grads_vars)
            self.initializer = tf.global_variables_initializer()
            print "Critic Model took %f seconds to load..." % (time.time() - start)

        
        self.start()

    def start(self):
        self._s = tf.Session(graph=self.g)
        with self.g.as_default():
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self._s)   
        self._s.run(self.initializer)
        # self.variables = ray.experimental.TensorFlowVariables(self.loss, self._s, prefix=True)
        self.g.finalize()

    def stop(self):
        pass

    def get_loss(self, obs, discounted_rwds):
        batch_feed = { self._input: obs, \
                self._discounted_rwds: discounted_rwds }
        return self._s.run(self.loss, feed_dict=batch_feed)

    def get_value(self, obs):
        batch_feed = { self._input: obs }
        return self._s.run(self._value, feed_dict=batch_feed)

    def compute_gradients(self, obs, discounted_rwds):
        batch_feed = { self._input: obs, \
                        self._discounted_rwds: discounted_rwds }

                # self._discounted_rwds: np.asarray(discounted_rwds)[:, np.newaxis]  }
        return self._s.run([g[0] for g in self.grads_vars], feed_dict=batch_feed)

    def model_update(self, grads):
        feed_dict = {self.grads_vars[i][0]: grads[i] 
                            for i in range(len(grads))}
        self._s.run(self._apply_gradients, feed_dict=feed_dict)

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        self.variables.set_weights(weights)