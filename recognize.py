import pickle
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

with open('resources/leaned_weights.pkl', 'rb') as f:
    network.params = pickle.load(f)


for idx in range(1000, 1010):
    img = x_train[idx]
    pred = network.predict(img)
    pred = np.argmax(pred)
    print(pred)

    label = t_train[idx]
    label = np.argmax(label)
    print(label)
    print()
