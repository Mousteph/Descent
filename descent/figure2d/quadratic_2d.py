import numpy as np

from .figure_2d import Figure2D

class Quadratic2D(Figure2D):
    def __name__(self):
        return f"Quadratic: gamma {self.gamma}"

    def __init__(self, gamma: float = 1):
        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        return self.gamma * (x**2) + x + 1