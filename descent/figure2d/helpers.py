from typing import Dict

X_AXIS_LABEL = "x axis"
Y_AXIS_LABEL = "y axis"
Z_AXIS_LABEL = "z axis"

def format_figure_2d(ax, parameters: Dict = {}):
    ax.grid(color='grey', linewidth=0.4, alpha=0.3, zorder=0)
    
    ax.set_title(parameters.get("title", "Dimension 1"))
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)

    y_lim = parameters.get('y_lim')
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    
    ax.legend()

    return ax