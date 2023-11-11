import numpy as np
from .figure_3d import Figure3D

class Quadratic3D(Figure3D):
    def __name__(self) -> str:
        """
        Returns the name of the class with gamma value.

        Returns:
            str: The name of the class with gamma value.
        """

        return f"Quadratic3D: gamma({self.gamma})"

    def __init__(self, gamma: np.array = np.array([1, 1])):
        """
        Initializes the Quadratic3D with the given gamma.

        Args:
            gamma (np.array, optional): The gamma values. Defaults to np.array([1, 1]).
        """
        
        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        """
        Returns the sum of gamma * x^2, x, and 1 along the last axis.
        R^2 -> R
        f(x) = gamma * x^2 + x + 1
        
        Args:
            x (np.array): The input array.

        Returns:
            np.array: The sum of gamma * x^2, x, and 1 along the last axis.
        """
        
        return (self.gamma * (x**2) + x + 1).sum(axis=-1)