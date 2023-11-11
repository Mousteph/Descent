from .gradient_descent import GradientDescent
from typing import Callable
import numpy as np

class GradientDescentConstant(GradientDescent):
    def __name__(self) -> str:
        """
        Returns the name of the class with the last mu value.

        Returns:
            str: The name of the class with the last mu value.
        """

        return f"{self.__class__.__name__}({self.__last_mu})"
        
    def __call__(self, f: Callable[[np.array], float], pk: np.array, 
                 mu: float = 0.001, eps: float = 1E-6,
                 max_iter: int = 10000, detect_div: float = 10e5) -> np.array:
        """
        Performs the gradient descent with a constant step size.

        Args:
            f (Callable[[np.array], float]): The function to minimize.
            pk (np.array): The initial point.
            mu (float, optional): The step size. Defaults to 0.001.
            eps (float, optional): The precision. Defaults to 1E-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 10000.
            detect_div (float, optional): The divergence detection value. Defaults to 10e5.

        Returns:
            np.array: The array of points visited during the gradient descent.
        """

        self.__last_mu = mu
        
        pk1 = pk - mu * self.gradient(f, pk)
        l = [pk]
        i = 0
        
        norm = np.linalg.norm(pk1 - pk, 2)
        
        while i < max_iter and (norm >= eps and norm <= detect_div):            
            pk, pk1 = pk1, pk1 - mu * self.gradient(f, pk1)
            norm = np.linalg.norm(pk1 - pk, 2)
            
            l.append(pk)
            i += 1

        l = np.array(l)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(l[-1]), i)
        
        return l