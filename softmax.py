import numpy as np


# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a)
#     return exp_a / sum_exp_a


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
