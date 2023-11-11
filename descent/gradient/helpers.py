import numpy as np

def partial(f, x: np.array, i: int = 0, dx: float = 1e-8) -> np.array:
    h = np.zeros(x.size)
    h[i] = dx
    
    return (f(x + h) - f(x - h)) / (2 * dx)


def gradient(f, x: np.array, dx: float = 1e-8):
    return np.array([partial(f, x, i, dx) for i in range(x.shape[0])])