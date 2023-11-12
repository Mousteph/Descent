import numpy as np
from typing import Callable

def partial(f: Callable[[np.array], float], x: np.array, i: int = 0, dx: float = 1e-8) -> np.array:
    """
    Calculates the partial derivative of the function at the given point.

    f'(x) = (f(x + h) - f(x - h)) / (2 * h)

    Args:
        f (Callable[[np.array], float]): The function to differentiate.
        x (np.array): The point at which to differentiate.
        i (int, optional): The index of the variable to differentiate. Defaults to 0.
        dx (float, optional): The small change in the variable. Defaults to 1e-8.

    Returns:
        np.array: The partial derivative of the function at the given point.
    """

    h = np.zeros(x.size)
    h[i] = dx
    
    return (f(x + h) - f(x - h)) / (2 * dx)


def gradient(f: Callable[[np.array], float], x: np.array, dx: float = 1e-8) -> np.array:
    """
    Calculates the gradient of the function at the given point.

    Args:
        f (Callable[[np.array], float]): The function to differentiate.
        x (np.array): The point at which to differentiate.
        dx (float, optional): The small change in the variable. Defaults to 1e-8.

    Returns:
        np.array: The gradient of the function at the given point.
    """

    return np.array([partial(f, x, i, dx) for i in range(x.shape[0])])