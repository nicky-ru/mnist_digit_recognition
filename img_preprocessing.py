import cv2
import numpy as np


def get_img(img=0):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (28, 28))

    cv2.imshow('orig', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = np.reshape(img, (784,))
    img = img / 255
    return img


