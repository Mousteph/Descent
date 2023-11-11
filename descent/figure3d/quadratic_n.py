import numpy as np

from .figure_3d import Figure3D


class QuadraticN(Figure3D):
    def __name__(self) -> str:
        return f"QuadraticN({self.A.shape[0]})"
    
    def __init__(self, A: np.array, b: np.array):
        self.A = A
        self.b = b
    
    def function(self, x: np.array) -> np.array:
        res = (x.T @ self.A @ x) / 2 - self.b.T @ x
        return res