import numpy as np

from .figure_3d import Figure3D


class Cubic3D(Figure3D):
    def __name__(self) -> str:
        return f"Cubic: gamma {self.gamma}"
    
    def __init__(self, gamma: np.array = np.array([1, 1])):
        self.gamma = gamma
        
    def function(self, x: np.array) -> np.array:
        return (x**3 + self.gamma * x**2 + x + 1).sum(axis=-1)