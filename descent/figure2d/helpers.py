from typing import Dict
from matplotlib.axes import Axes

X_AXIS_LABEL = "x axis"
Y_AXIS_LABEL = "y axis"

def format_figure_2d(ax: Axes, parameters: Dict[str, str] = {}) -> Axes:
    """
    Formats the 2D figure with the given parameters.

    Args:
        ax (Axes): The axes of the figure to format.
        parameters (Dict[str, str], optional): The parameters to use for formatting. 
            The parameters can include "title" for the title of the figure. Defaults to {}.

    Returns:
        Axes: The formatted axes.
    """

    ax.grid(color='grey', linewidth=0.4, alpha=0.3, zorder=0)
    
    ax.set_title(parameters.get("title", "Dimension 1"))
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)

    ax.legend()

    return ax