from src.function_figure import FunctionFigure

from typing import Dict
import numpy as np

class Quadratic(FunctionFigure):
    def function(self, x: np.array, gamma: float) -> np.array:
        return gamma * (x**2) + x + 1

    def plot_ax_2d(self, ax, parameters: Dict):
        x = parameters.get('x')
        gamma_range = parameters.get('gamma_range')
        
        for gamma in gamma_range:
            ax.plot(x, self.function(x, gamma), label="Gamma: {}".format(gamma))

        return ax
 
    def plot_ax_contour(self, ax, parameters: Dict, descent: Dict = {}):
        X = parameters.get('X')
        Y = parameters.get('Y')
        gamma = parameters.get('gamma')
        
        Z = self.function(X, gamma[0]) + self.function(Y, gamma[1])
        print(X.shape, Y.shape, Z.shape)
        ax.contour(X, Y, Z, 100, cmap='plasma', alpha=(1 if descent == {} else 0.5))

        if descent != {}:
            self.add_decent_2d_contour(ax, descent)
        return ax

    def plot_ax_3d(self, fig, ax, parameters: Dict, descent: Dict = {}):        
        X = parameters.get('X')
        Y = parameters.get('Y')
        gamma = parameters.get('gamma')
        
        Z = self.function(X, gamma[0]) + self.function(Y, gamma[1])

        image = ax.plot_surface(X, Y, Z, linestyles="solid", alpha=(1 if descent == {} else 0.3), cmap='plasma')
        fig.colorbar(image, shrink=0.5, aspect=10, pad=0.05)

        ax.contourf(X, Y, Z, zdir='z', offset=-2.5, cmap='plasma', alpha=(0.5 if descent == {} else 0.1))

        if descent != {}:
            self.add_descent_3d(ax, descent)

        return fig, ax