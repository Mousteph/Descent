import numpy as np

from .figure_3d import Figure3D


class Cubic3D(Figure3D):
    def __name__(self) -> str:
        """
        Returns the name of the class along with the value of gamma.

        Returns:
            str: The name of the class and the value of gamma.
        """

        return f"Cubic3D: gamma({self.gamma})"
    
    def __init__(self, gamma: np.array = np.array([1, 1])):
        """
        Initializes the Cubic3D class with a given gamma value.

        Args:
            gamma (np.array, optional): The gamma value. Defaults to np.array([1, 1]).
        """

        self.gamma = gamma
        
    def function(self, x: np.array) -> np.array:
        """
        Calculates the Cubic3D function for a given x.
        R^2 -> R
        f(x) = x^3 + gamma * x^2 + x + 1

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The output array after applying the Cubic3D function.
        """

        return (x**3 + self.gamma * x**2 + x + 1).sum(axis=-1)