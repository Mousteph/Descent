from .gd_optimal_step import GradientDescentOptimalStep
import numpy as np
from typing import Callable

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

        grad = self.gradient(f, x)
        dk = np.zeros(grad.shape[0])
        i = np.argmax(np.abs(grad))
        dk[i] = -grad[i]

        return dk

    def __call__(self, f: Callable[[np.array], float], pk: np.array, eps: float = 1E-6,
                 max_iter: int = 10000, detect_div: float = 10e5) -> np.array:
        """
        Performs the gradient descent with L1 optimization.

        Args:
            f (Callable[[np.array], float]): The function to minimize.
            pk (np.array): The initial point.
            eps (float, optional): The precision. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection value. Defaults to 10e5.

        Returns:
            np.array: The array of points visited during the gradient descent.
        """

        dk = self.dsgd(f, pk)
        pk1 = pk + self.backtrack(pk, f, dk) * dk
        l = [pk, pk1]
        i = 0
        
        norm = np.linalg.norm(pk1 - pk, 2)
        
        while i < max_iter and (norm >= eps and norm <= detect_div):
            pk = pk1
            dk = self.dsgd(f, pk)
            pk1 = pk + self.backtrack(pk, f, dk) * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            l.append(pk)
            i += 1
            
        l = np.array(l)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(l[-1]), i)
            
        return l