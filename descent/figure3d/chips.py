from .figure_3d import Figure3D
import numpy as np

class Chips(Figure3D):
    def __init__(self, a: float = 1, b: float = 1, h: float = 30):
        self.a = a
        self.b = b
        self.h = h
    
    def function(self, x: np.array) -> np.array:
        """
        Calculates the Chips function for a given x.
        R^2 -> R
        f(x, y) = (x^2 / a^2) - (y^2 / b^2) + h

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The output array after applying the Chips function.
        """

        return (x[0]**2) / (self.a ** 2) - (x[1] ** 2) / (self.b ** 2) + self.h