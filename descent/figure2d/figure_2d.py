import numpy as np
import matplotlib.pyplot as plt

from typing import Dict

from .helpers import format_figure_2d


class Figure2D:
    def __name__(self):
        return "Figure2D"
    
    def function(self, x: np.array) -> np.array:
        return x
    
    def plot_figure(self, ax, x: np.array = np.linspace(-5, 5, 400), descent: Dict = {}):
        alpha = 1 if descent == {} else 0.3
        
        ax.plot(x, self.function(x), label=self.__name__(), alpha=alpha)
        
        for key, values in descent.items():
            ax.plot(values, self.function(values), label=key, alpha=0.6, marker='x')
        
        return ax
    
    def figure(self, x: np.array = np.linspace(-5, 5, 400), descent: Dict = {}):
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax = self.plot_figure(ax, x, descent)
        ax = format_figure_2d(ax)
        
        plt.show()