import sys
import os
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict

sys.path.append(os.pardir)


class MultiLayerNet:
    def __init__(self):
        self.params = dict()
        self.layers = OrderedDict()
        self.last_layer = None

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

    def numerical_gradient(self, x, t):
        self.loss_W = lambda W: self.loss(x, t)


class TwoLayerNet(MultiLayerNet):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        super().__init__()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()

    def numerical_gradient(self, x, t):
        super().numerical_gradient(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(self.loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(self.loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(self.loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(self.loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        super().gradient(x, t)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


class FourLayerNet(MultiLayerNet):

    def __init__(self, input_size, first, second, third, output_size, weight_init_std=0.01):
        super().__init__()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, first)
        self.params['b1'] = np.zeros(first)
        self.params['W2'] = weight_init_std * np.random.randn(first, second)
        self.params['b2'] = np.zeros(second)
        self.params['W3'] = weight_init_std * np.random.randn(second, third)
        self.params['b3'] = np.zeros(third)
        self.params['W4'] = weight_init_std * np.random.randn(third, output_size)
        self.params['b4'] = np.zeros(output_size)

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def numerical_gradient(self, x, t):
        super().numerical_gradient(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(self.loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(self.loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(self.loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(self.loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(self.loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(self.loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(self.loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(self.loss_W, self.params['b4'])

        return grads

    def gradient(self, x, t):
        super().gradient(x, t)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db

        return grads
