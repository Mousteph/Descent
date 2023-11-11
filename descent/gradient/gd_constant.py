from .gradient_descent import GradientDescent

import numpy as np


class GradientDescentConstant(GradientDescent):
    def __name__(self):
        return f"{self.__class__.__name__}({self.__last_mu})"
        
    def __call__(self, f, pk, mu=0.001, eps=1E-6, max_iter=10000, detect_div=10e5):
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