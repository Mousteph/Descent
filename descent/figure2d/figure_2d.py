import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Dict

from .helpers import format_figure_2d


class Figure2D:
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
        Returns the input array as is.

        Args:
            x (np.array): The input array.

        Returns:
            np.array: The same input array.
        """

        return x
    
    def plot_figure(self, ax: Axes, x: np.array = np.linspace(-5, 5, 400), descent: Dict = {}) -> Axes:
        """
        Plots the figure on the given axes.

        Args:
            ax (Axes): The axes on which to plot the figure.
            x (np.array, optional): The input array. Defaults to np.linspace(-5, 5, 400).
            descent (Dict, optional): The descent values. Defaults to {}.

        Returns:
            Axes: The axes with the plotted figure.
        """

        alpha = 1 if descent == {} else 0.3
        
        ax.plot(x, self.function(x), label=self.__name__(), alpha=alpha)
        
        for key, values in descent.items():
            ax.plot(values, self.function(values), label=key, alpha=0.6, marker='x')
        
        return ax
    
    def figure(self, x: np.array = np.linspace(-5, 5, 400), descent: Dict = {}):
        """
        Plots the figure and shows it.

        Args:
            x (np.array, optional): The input array. Defaults to np.linspace(-5, 5, 400).
            descent (Dict, optional): The descent values. Defaults to {}.
        """

        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax = self.plot_figure(ax, x, descent)
        title = self.__name__()
        ax = format_figure_2d(ax, parameters={"title": title})
        
        plt.show()