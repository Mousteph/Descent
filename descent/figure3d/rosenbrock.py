import numpy as np
from .figure_3d import Figure3D

class Rosenbrock(Figure3D):
    def __name__(self) -> str:
        """
        Returns the name of the class with gamma value.

        Returns:
            str: The name of the class with gamma value.
        """

        return f"Rosenbrock: gamma({self.gamma})"
    
    def __init__(self, gamma: float = 1):
        """
        Initializes the Rosenbrock with the given gamma.

        Args:
            gamma (float): The gamma value. Defaults to 1.
        """
        
        self.gamma = gamma
        
    def function(self, x: np.array) -> np.array:
        """
        Returns the Rosenbrock function value for the given x.
        R^2 -> R
        f(x, y) = (x - 1)^2 + gamma * (x^2 - y)^2
        
        Args:
            x (np.array): The input array.

        Returns:
            np.array: The Rosenbrock function value.
        """
        
        return (x[0] - 1)**2 + self.gamma * (x[0]**2 - x[1])**2