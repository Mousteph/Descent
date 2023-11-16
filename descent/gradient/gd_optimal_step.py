from .gradient_descent import GradientDescent
import numpy as np
from typing import Callable
from .helpers import gradient


class GradientDescentOptimalStep(GradientDescent):
    def backtrack(self, x0: np.array, f: Callable[[np.array], float], dk: np.array,
                  alpha: float = 0.4, beta: float = 0.8, max_iteration:int = 80) -> float:
        """
        Performs the backtracking line search.

        Args:
            x0 (np.array): The current point.
            f (Callable[[np.array], float]): The function to minimize.
            dk (np.array): The descent direction.
            alpha (float, optional): The alpha value. Defaults to 0.4.
            beta (float, optional): The beta value. Defaults to 0.8.
            max_iteration (int, optional): The maximum number of iterations. Defaults to 80.

        Returns:
            float: The optimal step size.
        """

        mu = 1
        i = 0
        
        fx0 = f(x0)
        grad = dk.T @ gradient(f, x0)
        
        while f(x0 + mu * dk) >= (fx0 + alpha * mu * grad) and i < max_iteration:
            mu = mu * beta
            i += 1
        
        return mu
    
    @GradientDescent.calculate_time
    def __call__(self, f: Callable[[np.array], float], x0: np.array, eps: float = 1E-6,
                 max_iter: int = 10000, detect_div: float = 10e5) -> np.array:
        """
        Performs the gradient descent with optimal step size.

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
            dk = -gradient(f, pk)
            lr = self.backtrack(pk, f, dk)
            pk1, pk = pk, pk + lr * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            points.append(pk)
            i += 1

        points = np.array(points)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(points[-1]), i)
        
        return points