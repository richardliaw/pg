import numpy as np
from misc import sigmoid, discounted_cumsum, normalize

class KarpathyNN():
    """Personal re-implementation guided by Andrej Karpathy"""
    def __init__(self, input_size, hid_size, output_size=1):
        W1 = np.random.randn(hid_size, input_size)
        W2 = np.random.randn(hid_size)
        self.weights = {'W1': W1, 'W2':W2}
        self.dweights = {'W1': np.zeros_like(W1), 'W2':np.zeros_like(W2)}
        self.discount = 0.995
        self.alpha = 1e-2 # change to rms prop?
        self.reset_hidden()

    def reset_hidden(self):
        self.eph = {'h1': [], 'h2': [], 'dlogp': []} #hidden layer inputs


    def feedforward(self, I):
        """Returns action taken"""
        self.eph['h1'].append(I)
        m1 = np.dot(self.weights['W1'], I)
        m1[m1 < 0] = 0 # nonlinearity
        self.eph['h2'].append(m1)
        logp = np.dot(m1, self.weights['W2']) # not exactly logp but ok
        aprob = sigmoid(logp)
        action = int(np.random.uniform() < aprob) 
        self.add_dlogp(action, aprob)
        return action

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
        # discounted = normalize(discounted)
        for i in range(len(trajrwd)):
            grad = self.backprop(discounted[i], self.eph['h1'][i], self.eph['h2'][i], self.eph['dlogp'][i])
            for model in grad:
                self.dweights[model] += grad[model]
        self.reset_hidden()
    
    def model_update(self):
        for model in self.weights:
            # print("%s: " % model, self.weights[model][:4])
            self.weights[model] += self.alpha * self.dweights[model] # .. didn't average but it's ok
            self.dweights[model] = np.zeros_like(self.dweights[model])



