"""graph_tools"""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from helpers.consts import EVE_COLORS
from helpers.road_mesure import START, END

def init_single_column_plt(nb_graphes) :
    """Initialise un graphe a une seule colonne"""
    fig, axes = plt.subplots(
        nrows=nb_graphes,
        ncols=1,
        figsize=(15,6),
        sharex=True,
        gridspec_kw={'hspace': 0.8}
    )
    plt.rcParams.update({'font.size':5})
    return fig, axes


def habille(
    ax: Axes,
    scale: int,
    title: str | None = None,
    label: str | None = None,
    grid: bool = False
):
    """Habillage d'un axe"""
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, scale)
    if title:
        ax.set_title(title)
    if label:
        ax.set_ylabel(label, fontsize=8)
    if grid:
        ax.grid(visible=True, axis="x", linestyle="--")
        ax.grid(visible=True, axis="y")


def draw_object(
    label: str,
    x_pos: float,
    ymax: float,
    ax : Axes | None = None
) -> None:
    """Ajoute un évènement ponctuel
    et une ligne verticale de repérage.
    """
    current = ax if ax else plt
    current.annotate(
        label,
        (x_pos, 0.9 * ymax),
        bbox = EVE_COLORS[label]["bbox"],
        color = EVE_COLORS[label]["font"]
    )
    current.vlines(
        x_pos,
        0,
        1.5 * ymax,
        color = EVE_COLORS[label]["line"]
    )


def draw_objects(
    tops : dict[str, tuple],
    ymax: int,
    ax : Axes | None = None
):

    """Ajoute des évènements topés par le mesureur à un SI.
    tops est un dictionnaire avec 
    - clé = chaine topée
    - valeur = tuple (x,y) de la position sur le SI
    """
    for key, value in tops.items():
        x , _ = value
        if key == START:
            draw_object("D", x, ymax, ax=ax)
        elif key == END:
            draw_object("F", x, ymax, ax=ax)
        else:
            draw_object(key, x, ymax, ax=ax)
