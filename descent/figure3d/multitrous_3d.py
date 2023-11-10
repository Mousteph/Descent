import numpy as np

from .figure_3d import Figure3D


class Multitrous3D(Figure3D):
    def __name__(self) -> str:
        return f"Multitrous: gamma {self.gamma}"
    
    def __init__(self, gamma: np.array = np.array([1, 1])):
        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        return (20 * np.cos(x**2) + (self.gamma * x**2)).sum(axis=-1)