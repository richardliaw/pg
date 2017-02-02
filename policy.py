import numpy as np
from model import KarpathyNN
from misc import *

class Policy():
    def __init__(self, env, model_cls=KarpathyNN):
        self._model = model_cls(len(env.observation_space.high), 64)
        pass

    def get_action(self, state):
        action = self._model.feedforward(state)
        return action

    def process_gradient(self, trajrwd):
        self._model.policy_grad(trajrwd)

    def update(self, **kwargs):
        self._model.model_update(**kwargs)