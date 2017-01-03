import numpy as np
from misc import sigmoid, discounted_cumsum

class RLNN():
    def __init__(self, input_size, hid_size, output_size=1):
        W1 = np.random.randn(hid_size, input_size)
        W2 = np.random.randn(hid_size)
        self.weights = {'W1': W1, 'W2':W2}
        self.dweights = {'W1': np.zeros_like(W1), 'W2':np.zeros_like(W2)}
        self.discount = 0.995
        self.alpha = 1e-5 # change to rms prop?
        self.reset_hidden()

    def reset_hidden(self):
        self.eph = {'h1': [], 'h2': [], 'dlogp': []} #hidden layer inputs


    def feedforward(self, I):
        self.eph['h1'].append(I)
        m1 = np.dot(self.weights['W1'], I)
        m1[m1 < 0] = 0 # nonlinearity
        self.eph['h2'].append(m1)
        logp = np.dot(m1, self.weights['W2']) # not exactly logp but ok
        out = sigmoid(logp)
        return out

    def backprop(self, rwd, h1, h2, dlogp): #h1 is input
        dEdh = dlogp * self.weights['W2']
        dEdh[h2 <= 0] = 0 # backprop relu
        dW1 = rwd * np.outer(dEdh, h1) 
        dW2 = rwd * (dlogp * h2)
        return {'W2':dW2, 'W1': dW1}

    def add_dlogp(self, action, aprob):
        self.eph['dlogp'].append(action - aprob) # whatever this does

    def policy_grad(self, trajrwd):
        discounted = discounted_cumsum(trajrwd, self.discount)
        for i in range(len(trajrwd)):
            grad = self.backprop(discounted[i], self.eph['h1'][i], self.eph['h2'][i], self.eph['dlogp'][i])
            for model in grad:
                self.dweights[model] += grad[model]
        self.reset_hidden()
    
    def model_update(self):
        for model in self.weights:
            self.weights[model] += self.alpha * self.dweights[model]
            self.dweights[model] = np.zeros_like(self.dweights[model])

class Policy():
    def __init__(self, env):
        self._model = RLNN(len(env.observation_space.high), 64)
        pass

    def get_action(self, state):
        aprob = self._model.feedforward(state)
        action = int(np.random.uniform() < aprob)
        self._model.add_dlogp(action, aprob)
        return action

    def process_gradient(self, trajrwd):
        self._model.policy_grad(trajrwd)

    def update(self):
        self._model.model_update()


