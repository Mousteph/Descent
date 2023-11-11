import numpy as np

from .figure_2d import Figure2D


class Multitrous2D(Figure2D):
    def __name__(self) -> str:
        return f"Multitrous2D: gamma({self.gamma})"
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def function(self, x: np.array) -> np.array:
        return 20 * np.cos(x**2) + (self.gamma * x**2)