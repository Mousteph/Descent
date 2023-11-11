import numpy as np

from .figure_3d import Figure3D


class Rosenbrock(Figure3D):
    def __name__(self) -> str:
        return f"Rosenbrock: gamma({self.gamma})"
    
    def __init__(self, gamma: float):
        self.gamma = gamma
        
    def function(self, x: np.array) -> np.array:
        return (x[0] - 1)**2 + self.gamma * (x[0]**2 - x[1])**2