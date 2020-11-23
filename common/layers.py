import numpy as np
from .functions import softmax, cross_entropy, sigmoid


class Layer:
    def forward(self, x, t=0):
        pass

    def forward_loss(self, x, t):
        pass

    def backward(self, dout):
        pass


class Relu(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x, t=0):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x, t=0):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine(Layer):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x, t=0):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.loss, self.y, self.t = None, None, None

    def forward(self, x, t=0):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
