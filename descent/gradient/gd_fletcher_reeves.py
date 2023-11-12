from .gd_optimal_step import GradientDescentOptimalStep
import numpy as np


class GradientDescentFletcherReeves(GradientDescentOptimalStep):
    def gradient_compose_dk_FR(self, f, xk, x0, dk):
        grad = self.gradient(f, xk)
    
        return -grad + (np.linalg.norm(grad) ** 2 / np.linalg.norm(self.gradient(f, x0))**2) * dk
    
    def __call__(self, f, pk, eps=1E-6, max_iter=10000, detect_div=10e5, error=False):
        dk = -self.gradient(f, pk)
        pk1 = pk + self.backtrack(pk, f, dk) * dk
        l = [pk, pk1]
        i = 0
        
        norm = np.linalg.norm(pk1 - pk, 2)
        
        while i < max_iter and (norm >= eps and norm <= detect_div):
            dk = self.gradient_compose_dk_FR(f, pk1, pk, dk)
            pk = pk1
            pk1 = pk + self.backtrack(pk, f, dk) * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            l.append(pk)
            i += 1
            
        l = np.array(l)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(l[-1]), i)
        
        return l