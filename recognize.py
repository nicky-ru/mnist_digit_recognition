import pickle
from common.nets import FourLayerNet
from img_preprocessing import get_img
import numpy as np

img = get_img('dataset/WechatIMG205.jpeg')

network = FourLayerNet(input_size=784, first=100, second=100, third=50, output_size=10)

with open('resources/leaned_weights.pkl', 'rb') as f:
    network_to_unpickle = pickle.load(f)
    network.params = network_to_unpickle.get('params')
    network.layers = network_to_unpickle.get('layers')
    network.last_layer = network_to_unpickle.get('last_layer')

res = network.predict(img)
res = np.argmax(res)
print(res)
