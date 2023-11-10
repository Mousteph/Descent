import numpy as np

from .figure_2d import Figure2D


class Cubic2D(Figure2D):
    def __name__(self) -> str:
        return f"Cubic: gamma {self.gamma}"
        
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        
    def function(self, x: np.array) -> np.array:
        return x**3 + self.gamma * x**2 + x + 1