from .gd_optimal_step import GradientDescentOptimalStep
import numpy as np
from .helpers import gradient
from typing import Callable


class GradientDescentFletcherReeves(GradientDescentOptimalStep):
    def __init__(self):
        super().__init__()
        self.__last_grad = (None, None)

    def __get_last_grad(self, f: Callable[[np.array], float], xk: np.array) -> np.array:
        """A function to get the last gradient.

        Args:
            f (Callable[[np.array], float]): Function to compute the gradient.
            xk (np.array): Value where the gradient is computed.

        Returns:
            np.array: The gradient.
        """
        
        if np.array_equal(xk, self.__last_grad[0]):
            return self.__last_grad[1]
        
        return gradient(f, xk)
        
    def gradient_compose_dk_FR(self, f: Callable[[np.array], float], xk: np.array,
                               x0: np.array, dk: np.array) -> np.array:
        """
        Compute the gradient composition for Fletcher-Reeves method.

        Args:
            f (Callable[[np.array], float]): The function to optimize.
            xk (np.array): The current point in the optimization process.
            x0 (np.array): The initial point in the optimization process.
            dk (np.array): The current direction in the optimization process.

        Returns:
            np.array: The new direction for the next step in the optimization process.
        """

        grad = gradient(f, xk)
        grad_x0 = self.__get_last_grad(f, x0)
        dk = -grad + (np.linalg.norm(grad) ** 2 / np.linalg.norm(grad_x0)**2) * dk

        self.__last_grad = (xk, grad)
        
        return dk
    
    def __call__(self, f: Callable[[np.array], float], x0: np.array, eps: float = 1E-6,
                 max_iter: int = 10000, detect_div: float = 10e5):
        """
        Perform the optimization process using the Fletcher-Reeves method.

        Args:
            f (Callable[[np.array], float]): The function to optimize.
            x0 (np.array): The initial point in the optimization process.
            eps (float, optional): The precision of the solution. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection threshold. Defaults to 10e5.

        Returns:
            np.array: The points visited during the optimization process.
        """

        dk = -gradient(f, x0)
        mu = self.backtrack(x0, f, dk)
        pk1, pk = x0, x0 + mu * dk
        
        points = [pk1, pk]
        i = 0
        
        norm = np.linalg.norm(pk1 - pk, 2)
        
        while i < max_iter and (eps <= norm <= detect_div):
            dk = self.gradient_compose_dk_FR(f, pk, pk1, dk)
            mu = self.backtrack(pk, f, dk)
            pk1, pk = pk, pk + mu * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            points.append(pk)
            i += 1
            
        points = np.array(points)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(points[-1]), i)
        
        return points