from .gradient_descent import GradientDescent
import numpy as np


class GradientDescentOptimalStep(GradientDescent):
    def backtrack(self, x0: np.array, f , dir_x, alpha = 0.4, beta = 0.8, max_iteration=80):
        mu = 1
        i = 0
        
        fx0 = f(x0)
        grad = dir_x.T @ self.gradient(f, x0)
        
        while f(x0 + mu * dir_x) >= (fx0 + alpha * mu * grad) and i < max_iteration:
            mu = mu * beta
            i += 1
        
        return mu
    
    def __call__(self, f, pk, eps=1E-6, max_iter=10000, detect_div=10e5, error=False):    
        dk = -self.gradient(f, pk)
        
        pk1 = pk + self.backtrack(pk, f, dk) * dk
        l = [pk, pk1]
        i = 0
        
        norm = np.linalg.norm(pk1 - pk, 2)
        
        while i < max_iter and (norm >= eps and norm <= detect_div):
            pk = pk1
            dk = -self.gradient(f, pk)
            pk1 = pk + self.backtrack(pk, f, dk) * dk
            
            norm = np.linalg.norm(pk1 - pk, 2)
            l.append(pk)
            i += 1
            

        l = np.array(l)
        
        self._check_max_iter(i, max_iter)
        self._check_norm(norm, detect_div)
        self._set_report(f(l[-1]), i)
            
        
        return l