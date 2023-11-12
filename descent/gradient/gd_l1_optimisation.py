from .gd_optimal_step import GradientDescentOptimalStep
import numpy as np
from typing import Callable
from .helpers import gradient


class GradientDescentL1Optimisation(GradientDescentOptimalStep):
    def dsgd(self, f: Callable[[np.array], float], x: np.array) -> np.array:
        """
        Calculates the descent direction for the given function and point.

        Args:
            f (Callable[[np.array], float]): The function to minimize.
            x (np.array): The current point.

        Returns:
            np.array: The descent direction.
        """

        grad = gradient(f, x)
        dk = np.zeros(grad.shape[0])
        i = np.argmax(np.abs(grad))
        dk[i] = -grad[i]

        return dk

    def __call__(self, f: Callable[[np.array], float], x0: np.array, eps: float = 1E-6,
                 max_iter: int = 10000, detect_div: float = 10e5) -> np.array:
        """
        Performs the gradient descent with L1 optimization.

        Args:
            f (Callable[[np.array], float]): The function to minimize.
            x0 (np.array): The initial point.
            eps (float, optional): The precision. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection value. Defaults to 10e5.

        Returns:
            np.array: The array of points visited during the gradient descent.
        """
        
        pk, norm = x0, eps
        
        points = [x0]
        i = 0
        
        while i < max_iter and (eps <= norm <= detect_div):
            dk = self.dsgd(f, pk)
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