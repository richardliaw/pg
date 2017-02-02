
import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout
import time
from tfmodel import TFModel
from critic import Critic
import ray


class PolicyGradientAgent(TFModel):

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
                    weights_initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.contrib.layers.fully_connected(
                    inputs=hidden1,
                    num_outputs=hparams['num_actions'],
                    activation_fn=None)

            # op to sample an action - multinomial takes unnormalized log probs
            # logits = tf.Print(logits, [logits], "Logits - ")
            self._sample = tf.reshape(tf.multinomial(logits, 1), [])

            # get NORMALIZED log probabilities
            log_prob = tf.log(tf.nn.softmax(logits))

            # training part of graph
            self._acts = tf.placeholder(tf.int32)
            self._advantages = tf.placeholder(tf.float32)

            # get log probs of actions from episode
            indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

            loss = -tf.reduce_mean(tf.mul(act_prob, self._advantages))
            tf.summary.scalar("Actor Loss", loss)
            self.loss = loss
            # update
            self.optimizer = tf.train.GradientDescentOptimizer(hparams['learning_rate'])
            self._train = self.optimizer.minimize(loss)

            self.grads_vars = self.optimizer.compute_gradients(loss)
            self._apply_gradients = self.optimizer.apply_gradients(self.grads_vars)
            self.initializer = tf.global_variables_initializer()
            print "Policy Model took %f seconds to load..." % (time.time() - start)

        
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

    def act(self, observation):
        # get one action, by sampling
        return self._s.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self, obs, acts, advantages):
        batch_feed = { self._input: obs, \
                self._acts: acts, \
                self._advantages: advantages }
        self._s.run(self._train, feed_dict=batch_feed)

    def compute_gradients(self, obs, acts, advantages, debug=False):
        batch_feed = { self._input: obs, \
                self._acts: acts, \
                self._advantages: advantages }
        if debug:
            import ipdb; ipdb.set_trace()  # breakpoint 17a2637c //


        return self._s.run([g[0] for g in self.grads_vars], feed_dict=batch_feed)

    def model_update(self, grads):
        feed_dict = {self.grads_vars[i][0]: grads[i] 
                            for i in range(len(grads))}
        self._s.run(self._apply_gradients, feed_dict=feed_dict)

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        self.variables.set_weights(weights)