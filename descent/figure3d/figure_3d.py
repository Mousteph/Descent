import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Tuple, Dict

from .helpers import format_figure_3d, format_figure_contour_2d

class Figure3D:
    def __name__(self) -> str:
        """
        Returns the name of the class.

        Returns:
            str: The name of the class.
        """

        return self.__class__.__name__

    def __call__(self, x: np.array) -> np.array:
        """
        Calls the function method with the given x.

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The output array after applying the function.
        """
        
        return self.function(x)
    
    def function(self, x: np.array) -> np.array:
        """
        Returns the sum of the input array along the last axis.

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The sum of the input array along the last axis.
        """
        
        return x.sum(axis=-1)

    def calculate_xyz(self, x: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Calculates the X, Y, and Z values for the given x.

        Args:
            x (np.array): The input array.

        Returns:
            Tuple[np.array, np.array, np.array]: The X, Y, and Z values.
        """

        X, Y = np.meshgrid(x[:, 0], x[:, 1])
        Z = np.empty(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.function(np.array([X[i, j], Y[i, j]]))
        
        return X, Y, Z
    
    def plot_figure_3d(self, fig: Figure, ax: Axes, x: np.array = None,
                       descent: Dict = {}, shrink: int = 1) -> Tuple[Figure, Axes]:
        """
        Plots the 3D figure on the given axes.

        Args:
            fig (Figure): The figure on which to plot.
            ax (Axes): The axes on which to plot.
            x (np.array, optional): The input array. Defaults to None.
            descent (Dict, optional): The descent values. Defaults to {}.
            shrink (int, optional): The shrink factor for the colorbar. Defaults to 1.

        Returns:
            Tuple[Figure, Axes]: The figure and axes with the plotted 3D figure.
        """

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
            for key, (values, color) in descent.items():
                x_value = values[:, 0]
                y_value = values[:, 1]
                z_value = np.array([self.function(np.array([x_value[i], y_value[i]])) 
                                        for i in range(x_value.shape[0])])
                
                ax.plot(x_value, y_value, z_value, color=color, label=key, marker='x', markersize=1.3, linewidth=1)
            
            ax.legend()

        return fig, ax

    def plot_figure_contour(self, ax: Axes, x: np.array = None,
                            descent: Dict = {}) -> Tuple[Axes, Tuple[float, float]]:
        """
        Plots the contour figure on the given axes.

        Args:
            ax (Axes): The axes on which to plot.
            x (np.array, optional): The input array. Defaults to None.
            descent (Dict, optional): The descent values. Defaults to {}.

        Returns:
            Tuple[Axes, Tuple[float, float]]: The axes with the plotted contour figure and the Z limits.
        """

        if x is None:
            x = np.linspace(-5, 5, 100)
            x = np.stack((x, x), axis=-1)
        
        X, Y, Z = self.calculate_xyz(x) 
       
        alpha = 1 if descent == {} else 0.3 
        ax.contour(X, Y, Z, 100, cmap='plasma', alpha=alpha)
        
        if descent != {}:
            for key, (values, color) in descent.items():
                x_value = values[:, 0]
                y_value = values[:, 1]
                ax.plot(x_value, y_value, color=color, label=key, marker='x', markersize=1.3, linewidth=1)
            
            ax.legend()

        return ax, (Z.min(), Z.max())
    
    def figure(self, x: np.array = None, plot_3d: bool = True, plot_contour: bool = False,
               descent: Dict = {}, view: Tuple[int, int] = None):
        """
        Plots the figure with optional 3D and contour plots.

        Args:
            x (np.array, optional): The input array. Defaults to None.
            plot_3d (bool, optional): Whether to plot in 3D. Defaults to True.
            plot_contour (bool, optional): Whether to plot the contour. Defaults to False.
            descent (Dict, optional): The descent values. Defaults to {}.
            view (Tuple[int, int], optional): The view angles. Defaults to None.
        """
        
        colors = list(mcolors.TABLEAU_COLORS.keys())[:len(descent)]
        descent = {key: (values, colors[i]) for i, (key, values) in enumerate(descent.items())}
        
        fig = plt.figure(figsize=(12, 6))
        parameters = {
            "y_lim": (-5, 5) if x is None else (x[:, 1].min(), x[:, 1].max()),
            "x_lim": (-5, 5) if x is None else (x[:, 0].min(), x[:, 0].max()),
        }

        if plot_contour:
            ax = fig.add_subplot(1, 1 + plot_3d, 1)
            ax, Z = self.plot_figure_contour(ax, x, descent)
            parameters["title"] = f"Contour {self.__name__()}"
            parameters["z_lim"] = (Z[0], Z[1])
            ax = format_figure_contour_2d(ax, parameters=parameters)
        
        if plot_3d:
            ax = fig.add_subplot(1, 1 + plot_contour, 1 + plot_contour, projection='3d')
            fig, ax = self.plot_figure_3d(fig, ax, x, descent, shrink=(0.9 / (2 - plot_contour)))
            parameters["title"] = f"Surface {self.__name__()}"
            parameters["view"] = view
            ax = format_figure_3d(ax, parameters=parameters)

        plt.show()