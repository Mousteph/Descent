import numpy as np
from .figure_3d import Figure3D

class QuadraticN(Figure3D):
    def __name__(self) -> str:
        """
        Returns the name of the class with the shape of A.

        Returns:
            str: The name of the class with the shape of A.
        """

        return f"QuadraticN({self.A.shape[0]})"
    
    def __init__(self, a: np.array, b: np.array):
        """
        Initializes the QuadraticN with the given A and b.

        Args:
            a (np.array): The A values.
            b (np.array): The b values.
        """
        
        self.A = a
        self.b = b
    
    def function(self, x: np.array) -> np.array:
        """
        Returns the result of (x.T @ A @ x) / 2 - b.T @ x.
        R^n -> R
        f(x) = (x.T @ A @ x) / 2 - b.T @ x

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The result of (x.T @ A @ x) / 2 - b.T @ x.
        """

        res = (x.T @ self.A @ x) / 2 - self.b.T @ x
        return res