import numpy as np
import math


def create_system(dim, cond: int = 10, seed: int = 100):
    rng = np.random.default_rng(seed)
    cond_sqrt = math.sqrt(cond)
    
    A = 0.1 * rng.uniform(-cond_sqrt, cond_sqrt, size=(dim,dim))
    A = np.triu(A)
    
    # replace the diagonal of A with random positive values between 1 and sqrt(cond)
    A = A - np.diag(np.diag(A)) + np.diag(rng.uniform(1., cond_sqrt, size=(dim))) 
    
    # set the first two terms of the diagonal of A to fix the conditioning
    A[0,0] = 1.
    A[1,1] = cond_sqrt
    A = A.T @ A
    
    b = 1. * rng.integers(-10, 10, size=(dim))
    
    return A,b