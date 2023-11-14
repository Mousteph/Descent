import numpy as np
from typing import Callable
from .gradient_descent import GradientDescent
from .helpers import gradient


class GradientDescentAdam(GradientDescent):
    @GradientDescent.calculate_time
    def __call__(self, f: Callable[[np.array], float], x0: np.array, beta1: float = 0.9,
                 beta2: float = 0.999, alpha: float = 0.01, epszero: float = 1E-8,
                 eps: float = 1E-6, max_iter: int = 10000, detect_div: float = 10e5) -> np.array:
        """Performs the gradient descent with Adam.

        Args:
            f (Callable[[np.array], float]): The function to optimize.
            x0 (np.array): The initial point.
            beta1 (float, optional): The decay rate for the moving average of the gradient. Defaults to 0.9.
            beta2 (float, optional): The decay rate for the moving average of the squared gradient. Defaults to 0.999.
            alpha (float, optional): The learning rate. Defaults to 0.1.
            epszero (float, optional): The smoothing term to avoid division by zero. Defaults to 1E-8.
            eps (float, optional): The precision. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection threshold. Defaults to 10e5.

        Returns:
            np.array: The points visited during the optimization.
        """
        
        pk, norm = x0, eps
        mt = 0
        vt = 0
        points = [x0]
        i = 0
        
        while i < max_iter and (eps <= norm <= detect_div):
            grad = gradient(f, pk)
            mt = beta1 * mt + (1 - beta1) * grad
            vt = beta2 * vt + (1 - beta2) * grad**2
            
            mthat = mt / (1 - beta1**(i + 1))
            vthat = vt / (1 - beta2**(i + 1))
            dk = (mthat / (np.sqrt(vthat) + epszero))
            
            pk, pk1 = pk - alpha * dk, pk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            points.append(pk)
            i += 1
            
        points = np.array(points)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(points[-1]), i)
            
        return points