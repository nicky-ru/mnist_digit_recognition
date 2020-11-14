import sys
import os
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
sys.path.append(os.pardir)


def img_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)
img = x_train[1]
label = t_train[1]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
