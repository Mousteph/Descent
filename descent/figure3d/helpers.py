from typing import Dict, Optional
from matplotlib.axes import Axes

X_AXIS_LABEL = "x axis"
Y_AXIS_LABEL = "y axis"
Z_AXIS_LABEL = "z axis"

def format_figure_contour_2d(ax: Axes, parameters: Dict[str, Optional[str]] = {}) -> Axes:
    """
    Formats the 2D contour figure with the given parameters.

    Args:
        ax (Axes): The axes of the figure.
        parameters (Dict[str, Optional[str]], optional): The parameters for the figure. Defaults to {}.

    Returns:
        Axes: The formatted axes.
    """

    ax.grid(color='grey', linewidth=0.4, alpha=0.3, zorder=0)
    
    ax.set_title(parameters.get("title", "Contour Dimension 2"))
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)

    return ax

def format_figure_3d(ax: Axes, parameters: Dict[str, Optional[str]] = {}) -> Axes:
    """
    Formats the 3D figure with the given parameters.

    Args:
        ax (Axes): The axes of the figure.
        parameters (Dict[str, Optional[str]], optional): The parameters for the figure. Defaults to {}.

    Returns:
        Axes: The formatted axes.
    """

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_zlabel(Z_AXIS_LABEL)
    
    ax.xaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))
    ax.yaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))
    ax.zaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.5)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.5)

    ax.set_title(parameters.get("title", "Surface Dimension 2"))
        
    view = parameters.get('view')
    if view is not None:
        ax.view_init(*view)

    return ax