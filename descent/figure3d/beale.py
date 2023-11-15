from .figure_3d import Figure3D
import numpy as np


class Beale(Figure3D):
    def function(self, x: np.array) -> np.array:
        """
        Calculates the Beale function for a given x.
        R^2 -> R
        f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The output array after applying the Beale function.
        """
        
        return (1.5 - x[0] + x[0]*x[1]) ** 2 + \
               (2.25 - x[0] + x[0]*x[1]) ** 2 + \
               (2.625 - x[0] + x[0]*x[1]**3) ** 2