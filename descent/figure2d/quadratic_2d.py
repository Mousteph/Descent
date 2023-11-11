import numpy as np

from .figure_2d import Figure2D


class Quadratic2D(Figure2D):
    def __name__(self) -> str:
        """
        Returns the name of the class along with the value of gamma.

        Returns:
            str: The name of the class and the value of gamma.
        """

        return f"Quadratic2D: gamma({self.gamma})"

    def __init__(self, gamma: float = 1.0):
        """
        Initializes the Quadratic2D class with a given gamma value.

        Args:
            gamma (float, optional): The gamma value. Defaults to 1.0.
        """

        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        """
        Calculates the Quadratic2D function for a given x.
        f(x) = gamma * x^2 + x + 1

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The output array after applying the Quadratic2D function.
        """
        
        return self.gamma * (x**2) + x + 1