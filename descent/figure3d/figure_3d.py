import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict

from .helpers import format_figure_3d, format_figure_contour_2d

class Figure3D:
    def __name__(self):
        return "Figure3D"

    def __call__(self, x: np.array) -> np.array:
        return self.function(x)
    
    def function(self, x: np.array) -> np.array:
        return x.sum(axis=-1)

    def calculate_xyz(self, x: np.array) -> Tuple[np.array, np.array, np.array]:
        X, Y = np.meshgrid(x[:, 0], x[:, 1])
        Z = np.empty(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.function(np.array([X[i, j], Y[i, j]]))
        
        return X, Y, Z
    
    def plot_figure_3d(self, fig, ax, x: np.array = None, descent: Dict = {}, shrink: int = 1):
        if x is None:
            x = np.linspace(-5, 5, 100)
            x = np.stack((x, x), axis=-1)
        
        X, Y, Z = self.calculate_xyz(x) 
        
        alpha = 1 if descent == {} else 0.3
        image = ax.plot_surface(X, Y, Z, linestyles="solid", cmap='plasma', alpha=alpha)
        fig.colorbar(image, shrink=shrink, aspect=35, pad=0.05, orientation="horizontal")

        alpha = 0.4 if descent == {} else 0.2
        ax.contourf(X, Y, Z, zdir='z', offset=-2.5, cmap='plasma', alpha=alpha)
        
        if descent != {}:
            for key, values in descent.items():
                x_value = values[:, 0]
                y_value = values[:, 1]
                z_value = np.array([self.function(np.array([x_value[i], y_value[i]])) 
                                        for i in range(x_value.shape[0])])
                
                ax.plot(x_value, y_value, z_value, label=key, marker='x', markersize=1.3, linewidth=1)
        
        return fig, ax

    def plot_figure_contour(self, ax, x: np.array = None, descent: Dict = {}):
        if x is None:
            x = np.linspace(-5, 5, 100)
            x = np.stack((x, x), axis=-1)
        
        X, Y, Z = self.calculate_xyz(x) 
       
        alpha = 1 if descent == {} else 0.3 
        ax.contour(X, Y, Z, 100, cmap='plasma', alpha=alpha)
        
        if descent != {}:
            for key, values in descent.items():
                x_value = values[:, 0]
                y_value = values[:, 1]
                ax.plot(x_value, y_value, label=key, marker='x', markersize=1.3, linewidth=1)

        return ax
    
    def figure(self, x: np.array = None, plot_3d: bool = True, plot_contour: bool = False,
               descent: Dict = {}, view: Tuple[int, int] = None):
        fig = plt.figure(figsize=(12, 6))

        if plot_contour:
            ax = fig.add_subplot(1, 1 + plot_3d, 1)
            ax = self.plot_figure_contour(ax, x, descent)
            title = f"Contour {self.__name__()}"
            ax = format_figure_contour_2d(ax, parameters={"title": title})
        
        if plot_3d:
            ax = fig.add_subplot(1, 1 + plot_contour, 1 + plot_contour, projection='3d')
            fig, ax = self.plot_figure_3d(fig, ax, x, descent, shrink=(0.9 / (2 - plot_contour)))
            parameters = {
                "title": f"Surface {self.__name__()}",
                "view": view,
            }
            ax = format_figure_3d(ax, parameters=parameters)

        plt.show()