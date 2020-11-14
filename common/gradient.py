import numpy as np


def numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_value = x[idx]

        x[idx] = tmp_value + h
        fxh1 = f(x)

        x[idx] = tmp_value - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_value

    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, x in enumerate(x):
            grad[idx] = numerical_gradient_no_batch(f, x)
        return grad
