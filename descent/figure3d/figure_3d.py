import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from .helpers import format_figure_3d, format_figure_contour_2d

class Figure3D:
    def __name__(self):
        return "Figure3D"
    
    def function(self, x: np.array) -> np.array:
        return x.sum(axis=-1)

    def calculate_xyz(self, x: np.array) -> Tuple[np.array, np.array, np.array]:
        X, Y = np.meshgrid(x[:, 0], x[:, 1])
        Z = self.function(np.stack((X, Y), axis=-1))
        
        return X, Y, Z
    
    def plot_figure_3d(self, fig, ax, x: np.array = None):
        if x is None:
            x = np.linspace(-5, 5, 100)
            x = np.stack((x, x), axis=-1)
        
        X, Y, Z = self.calculate_xyz(x) 
        
        image = ax.plot_surface(X, Y, Z, linestyles="solid", cmap='plasma')
        fig.colorbar(image, shrink=0.5, aspect=10, pad=0.05)
        ax.contourf(X, Y, Z, zdir='z', offset=-2.5, cmap='plasma', alpha=0.4)
        
        return fig, ax

    def plot_figure_contour(self, ax, x: np.array = None):
        if x is None:
            x = np.linspace(-5, 5, 100)
            x = np.stack((x, x), axis=-1)
        
        X, Y, Z = self.calculate_xyz(x) 
        
        ax.contour(X, Y, Z, 100, cmap='plasma')

        return ax
    
    def figure(self, x: np.array = None, plot_3d: bool = True, plot_contour: bool = False):
        fig = plt.figure(figsize=(16, 6))

        if plot_contour:
            ax = fig.add_subplot(1, 1 + plot_3d, 1)
            ax = self.plot_figure_contour(ax, x)
            ax = format_figure_contour_2d(ax)
        
        if plot_3d:
            ax = fig.add_subplot(1, 1 + plot_contour, 1 + plot_contour, projection='3d')
            fig, ax = self.plot_figure_3d(fig, ax, x)
            ax = format_figure_3d(ax)

        plt.show()