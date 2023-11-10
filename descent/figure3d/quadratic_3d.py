import numpy as np

from .figure_3d import Figure3D


class Quadratic3D(Figure3D):
    def __name__(self):
        return f"Quadratic3D: gamma {self.gamma}"

    def __init__(self, gamma: np.array = np.array([1, 1])):
        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        return (self.gamma * (x**2) + x + 1).sum(axis=-1)