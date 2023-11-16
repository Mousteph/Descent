from typing import Callable, List
import numpy as np
from .gradient_descent import GradientDescent
from .helpers import gradient


class GradientDescentMomentumAcceleration(GradientDescent):
    @GradientDescent.calculate_time
    def __call__(self, f: Callable[[np.array], float], x0: np.array, lr: float = 0.001,
                 gamma: float = 0.9, eps: float = 1E-6, max_iter: int = 10000,
                 detect_div: float = 10e5) -> np.array:
        """Performs the gradient descent with momentum acceleration.

        Args:
            f (Callable[[np.array], float]): The function to optimize.
            x0 (np.array): The initial point.
            lr (float, optional): The learning rate. Defaults to 0.001.
            gamma (float, optional): The momentum factor. Defaults to 0.9.
            eps (float, optional): The precision. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection threshold. Defaults to 10e5.

        Returns:
            np.array: The points visited during the optimization.
        """
        
        pk, norm = x0, eps
        dk = np.zeros_like(x0)
        points = [x0]
        i = 0
        
        while i < max_iter and (eps <= norm <= detect_div):
            dk = gamma * dk + (1 - gamma) * gradient(f, pk)
            pk1, pk = pk, pk - lr * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            points.append(pk)
            i += 1
            
        points = np.array(points)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(points[-1]), i)
            
        return points