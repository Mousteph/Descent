import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from src.format_figures import format_figure_2d, format_figure_3d, format_figure_contour_2d

class FunctionFigure:        
    def figure_2d(self, func_params: Dict, figu_params: Dict):
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax = self.plot_ax_2d(ax, func_params)
        ax = format_figure_2d(ax, figu_params)
        
        plt.show()
    
    def add_descent_3d(self, ax, descent: Dict):
        for key, values in descent.items():
            Z = np.zeros((len(values)))
            for i in range(len(values)):
                Z[i] = self.function(values[i])

            ax.plot(values[:,0], values[:,1], Z, marker='x', markersize=1.3, linewidth=1, label=key)

        ax.legend()

        return ax     

    def figure_3d(self, func_params: Dict, figu_params: Dict, descent: Dict = {}):
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes(projection='3d')

        fig, ax = self.plot_ax_3d(fig, ax, func_params, descent)
        ax = format_figure_3d(ax, figu_params)

        plt.show()
        
    def add_decent_2d_contour(self, ax, descent: Dict):
        for key, values in descent.items():
            ax.plot(values[:,0], values[:,1], marker='x', markersize=2, linewidth=1.5, label=key)

        ax.legend()

        return ax 

    def contour_2d(self, func_params: Dict, figu_params: Dict, descent: Dict = {}):
        _, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax = self.plot_ax_contour(ax, func_params, descent)
        ax = format_figure_contour_2d(ax, figu_params)

        plt.show()

    def figures(self, func_2d_params: Dict, figu_2d_params: Dict, func_3d_params: Dict, figu_3d_params: Dict):
        fig = plt.figure(figsize=(16, 6))
  
        # First subplot
        ax = fig.add_subplot(1, 2, 1)
        ax = self.plot_ax_2d(ax, func_2d_params)
        ax = format_figure_2d(ax, figu_2d_params)
        
        # Second subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        fig, ax = self.plot_ax_3d(fig, ax, func_3d_params)
        ax = format_figure_3d(ax, figu_3d_params)

        plt.show()

    def figure_contour(self, func_3d_params: Dict, figu_3d_params: Dict, descent: Dict = {}):
        fig = plt.figure(figsize=(16, 6))
  
        # First subplot
        ax = fig.add_subplot(1, 2, 1)
        ax = self.plot_ax_contour(ax, func_3d_params, descent)
        ax = format_figure_contour_2d(ax, figu_3d_params)
        
        # Second subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        fig, ax = self.plot_ax_3d(fig, ax, func_3d_params, descent)
        ax = format_figure_3d(ax, figu_3d_params)

        plt.show()