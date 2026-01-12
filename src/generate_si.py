"""Traitement des fichiers du griptester MK2.
sous la forme de schémas itinéraires SI
"""
import argparse
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from helpers.consts import (
    UPPER, LOWER,
    LEVELS, LEGENDS,
    COLORS, get_color
)
from helpers.shared import pick_files, which_measure
from helpers.apo import get_apo_datas
from helpers.grip import get_grip_datas
from helpers.generic_absdatatop_csv import get_generic_absdatatop_csv
from helpers.road_mesure import RoadMeasure
from helpers.tools_file import CheckConf
from helpers.graph_tools import draw_objects, init_single_column_plt, habille
from helpers.iq3d import GraphStates
from helpers.consts_etat_surface import surface_state_legend



YAML_CONF = CheckConf()

PRECISION = {
    100: 0,
    1: 2
}

# pas en mètres pour une analyse en zône homogène
MEAN_STEP = YAML_CONF.get_mean_step()

# Mapping entre sens des mesures Grip et sens Aigle
SENS_GRIP_TO_AIGLE = {
    "D": "P",  # Droite → P
    "G": "M",  # Gauche → M
}

@dataclass
class Aigle :
    """aigle3D dataclass"""
    route = YAML_CONF.yaml.get("aigle_route")
    dep = YAML_CONF.yaml.get("aigle_dep")
    sens_list = YAML_CONF.yaml.get("aigle_sens", ["P"]) # par défaut "P" si rien
    recalage = YAML_CONF.yaml.get("aigle_recalage", {})
    df = None

aigle = Aigle()

def color_map(
    y_data: list[float],
    unit: str = "CFT"
)-> list[str]:
    """Crée le tableau des couleurs pour l'histogramme."""
    return [get_color(val, unit) for val in y_data]


def filtre_bornes(mes: RoadMeasure, bornes: list[str] | None):
    """Filtre les données de la mesure fonction des bornes fournies."""
    if not bornes or bornes is None:
        mes.clear_zoom()
    elif len(bornes) == 1:
        mes.apply_zoom_from_prs(bornes[0], None)
    elif len(bornes) >= 2:
        mes.apply_zoom_from_prs(bornes[0], bornes[-1])
    return mes.abs_zoomed(), mes.datas_zoomed


def get_measures(nb_mes) -> list[RoadMeasure]:
    """construit la liste des mesures."""
    questions = {}
    for j in range(nb_mes):
        questions[f"measure_{j}"] = {
            "folder_path" : YAML_CONF.get_datas_folder(),
            "ext": ["csv", "RE0"],
            "message": f"fichier de mesure {j}"
        }

    file_names = pick_files(
        **questions
    )
    measures: list[RoadMeasure] = []

    for name in file_names.values():
        mes_unit = which_measure(name)
        print(f"{name} > unité de mesure : {mes_unit}")
        force_sens = None
        if "droite" in name.lower():
            force_sens = "D"
        if "gauche" in name.lower():
            force_sens = "G"
        datas : RoadMeasure | None = None
        if mes_unit == "CFL":
            datas = get_grip_datas(name, force_sens=force_sens)
        if mes_unit == "PMP":
            datas = get_apo_datas(name, unit=mes_unit, force_sens=force_sens)
        if mes_unit == "CFT":
            datas = get_generic_absdatatop_csv(name, unit=mes_unit, force_sens=force_sens)
        if datas is not None:
            measures.append(datas)
    return measures


def format_legend(add_percent, unit, data):
    """Formate la légende, avec ou sans %"""
    family_counts = {}
    if add_percent :
        levels_description = LEVELS[unit]
        for level, bounds in levels_description.items():
            lower = bounds.get(LOWER, float("-inf"))
            upper = bounds.get(UPPER, float("inf"))
            family_counts[level] = sum(1 for v in data if lower < v <= upper)

    legend = []
    for level, color_label in LEGENDS[unit].items():
        if add_percent and level in family_counts:
            pct = 100 * family_counts[level] / len(data)
            legend_text = f"{color_label} ({pct:.1f}%)"
        else:
            legend_text = color_label  # affichage simple sans %
        patch = mpatches.Patch(
            color=COLORS[unit][level],
            label=legend_text
        )
        legend.append(patch)
    return legend


def draw_colored_horizons(
    unit: str,
    y_max: int,
    ax: Axes
):
    """Ajout de bandes colorées en arrière-plan"""
    if unit in ("CFT", "CFL"):
        for level, val in LEVELS[unit].items():
            lower = val.get(LOWER, 0)
            upper = val.get(UPPER, y_max)
            ax.axhspan(
                lower,
                upper,
                color=COLORS[unit][level],
                alpha=YAML_CONF.get_backgound_alpha(level)
            )


def draw_mean_histo(
    mes: RoadMeasure,
    y_max: int,
    rec_zh: str,
    ax : Axes
):
    """affiche l'histogramme des valeurs moyennes."""
    if not mes.unit:
        return
    x_mean_values, mean_values = mes.produce_mean(
        MEAN_STEP,
        rec_zh=rec_zh
    )
    ax.bar(
        x_mean_values,
        mean_values,
        width=MEAN_STEP,
        color=color_map(mean_values, unit=mes.unit),
        edgecolor="white"
    )
    for jj, mean_value in enumerate(mean_values):
        ax.annotate(
            round(mean_value, PRECISION[y_max]),
            (x_mean_values[jj], mean_value)
        )


def fix_abs_reference(measures: list[RoadMeasure], pr: str | None, grapher = None):
    """fixe l'abscisse de référence"""
    if pr is None:
        print("Pas de pr de recalage fourni")
        return None
    # Cas A3D présent
    grip_sens = measures[0].sens # D ou G
    aigle_sens = SENS_GRIP_TO_AIGLE.get(grip_sens)
    if grapher is not None:
        try:
            abs_reference = grapher.curv_prs[aigle_sens][pr]
            print(f"abscisse du pr A3D {pr}(Grip{grip_sens} -> aigle{aigle_sens}):{abs_reference}")
            return abs_reference
        except KeyError:
            print(f"Attention le PR A3D saisi '{pr}' est inexistant : pas de recalage")

    # Cas sans A3D
    try:
        abs_reference = measures[0].tops()[pr][0]
        print(f"abscisse du pr Grip {pr} dans cette mesure : {abs_reference}")
        return abs_reference
    except KeyError:
        print(f"Attention le PR Grip saisi '{pr}' est inexistant : pas de recalage")
        return None

def extract_prd_prf(args):
    """bornes pour prd/prf """
    if args.bornes:
        prd = int(args.bornes[0])
        prf = int(args.bornes[-1]) if len(args.bornes) > 1 else None
        return prd, prf

    if args.pr:
        return int(args.pr), None

    return None, None

def init_context(args):
    """Initialise le contexte de graphes (Aigle + matplotlib)."""
    grapher = None
    nb_graphes = 0
    if aigle.route and aigle.dep :
        grapher = GraphStates()

        grapher.set_route_dep(route=aigle.route, dep=aigle.dep)
        nb_graphes += 3 * len(aigle.sens_list)

    measures = get_measures(int(args.multi))

    nb_graphes += len(measures) if MEAN_STEP == 0 else 2*len(measures)
    _, axes = init_single_column_plt(nb_graphes)

    return grapher, measures, axes


def main(args):
    """main exe"""
    grapher, measures, axes = init_context(args)
    plt_index = 0
    if grapher:
        fig = axes[0].figure
        fig.legend(
            handles = surface_state_legend(),
            loc = "upper right",
            ncol = len(surface_state_legend()),
        )
        for sens in aigle.sens_list :
            prd, prf = extract_prd_prf(args)
            aigle.df = grapher.graphe_sens(
                sens=sens,
                axes=axes[plt_index:plt_index+3],
                prd=prd,
                prf=prf
            )
            plt_index += 3
    abs_reference = fix_abs_reference(
        measures,
        args.pr,
        grapher if (aigle.route and aigle.dep) else None
    )
    nb_sens_mono = len({mes.sens for mes in measures})
    for j, mes in enumerate(measures):
        y_max = 100 if mes.unit in  ("CFT","CFL") else 1
        print(f"mesure {j}")
        ax: Axes = axes[plt_index]
        habille(ax, y_max, title=mes.title, grid=True)

        print(f"tops avant offset {mes.tops()}")
        if j != 0 and mes.sens != measures[0].sens:
            mes.reverse()
        elif YAML_CONF.get("force_reverse") and nb_sens_mono == 1:
            mes.reverse()
        if abs_reference is not None:
            mes.offset = abs_reference - mes.tops()[args.pr][0]
            print(f""""
            on applique un offset {mes.offset}
            tops après offset : {mes.tops()}
            """)
        # Fusion de abscisse et data en abscisses_data
        # abscisses_data[0] vaut abscisses et [1] vaut data
        abscisses_data = filtre_bornes(mes, args.bornes)
        if args.bornes and j == 0:
            ax.set_xlim(min(abscisses_data[0]), max(abscisses_data[0]))
        n = len(abscisses_data[1])
        if n == 0:
            continue
        draw_colored_horizons(mes.unit, y_max, ax=ax)

        print(f"il y a {n} lignes")
        if mes.unit is None:
            continue
        if YAML_CONF.view_legend():
            ax.legend(
                handles=format_legend(
                    args.add_percent,
                    mes.unit,
                    abscisses_data[1]
                ),
                loc="upper right"
            )

        draw_objects(mes.tops(), y_max, ax=ax)
        ax.bar(
            abscisses_data[0],
            abscisses_data[1],
            width = mes.step,
            color = color_map(abscisses_data[1], unit=mes.unit),
            edgecolor = color_map(abscisses_data[1], unit=mes.unit)
        )
        plt_index += 1

        if MEAN_STEP :
            ax = axes[plt_index]
            habille(ax, y_max)
            draw_mean_histo(mes, y_max, args.rec_zh, ax=ax)
            draw_objects(mes.tops(), y_max, ax=ax)
            plt_index += 1
    return measures


def summarize(list_of_measures):
    """affiche des éléments synthétiques sur les mesures."""
    for index, measure in enumerate(list_of_measures):
        msg = f"""
        SUMMARY mesure {index}
        sens {measure.sens}
        offset {measure.offset}
        abscisse départ {measure.abs()[0]}
        abscisse fin {measure.abs()[-1]}
        """
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='linear diagrams')
    parser.add_argument(
        "--multi",
        action="store",
        help="nombre de csv à traiter",
        default=1
    )
    parser.add_argument(
        "--pr",
        action="store",
        help="pr de recalage",
        default=None
    )
    parser.add_argument(
        "--add_percent",
        action="store_true",
        help="Afficher la légende avec les pourcentages"
    )
    parser.add_argument(
        "--bornes",
        nargs = "*",
        default=None,
        help="Fixer manuellement les bornes d'affichage"
    )
    parser.add_argument(
        "--rec_zh",
        default=None,
        help="événement pour le recalage des zones homogènes"
    )
    summarize(main(parser.parse_args()))
    plt.show()
