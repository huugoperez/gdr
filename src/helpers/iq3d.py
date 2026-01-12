"""Helpers pour l'analyse des états de surface IQRN 3D"""
from collections import defaultdict
from itertools import accumulate

import pandas as pd
from pandas import DataFrame
from pandas import Series
from matplotlib.axes import Axes
from helpers.graph_tools import draw_object, habille

from helpers.consts_etat_surface import (
    FILE,
    COLORS,
    ABD, ABF,
    PLOD, PLOF,
    ROUTE, DEP, SENS,
    LONGUEUR_TRONCON,
    SURF_EVAL,
    PR_REGEX,
    STATES,IES,IEP,IETP,
    PRD_NUM, PRF_NUM, PRD, PRF, PRD_NAT,
    CURV_START, CURV_END,
    Y_SCALE, Y_SCALE_W_PR,
    D_SUP,NB_LEVELS,
    MESSAGE_NO_DF,
    level_name,
    pct_name
)

class SurfaceAnalyzer:
    """Classe pour analyser les états """
    def __init__(
        self,
        file_path: str | None = None,
        df : DataFrame | None = None
        ) -> None:
        """Initialisation"""
        self.file_path = file_path
        self.df = df
        self.sheet_name : str | None = None


    def load_sheet(self):
        """Charge une feuille"""
        assert self.file_path is not None
        excel_file = pd.ExcelFile(self.file_path)
        print("Feuilles disponibles dans le fichier :")
        for i, sheet in enumerate(excel_file.sheet_names):
            print(f"{i}: {sheet}")

        while True:
            sheet_index = int(input("Entrez le numéro de la feuille à charger : "))
            if 0 <= sheet_index < len(excel_file.sheet_names):
                break
            print("Numéro invalide, réessayez.")

        sheet_name = excel_file.sheet_names[sheet_index]
        self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        print(f"Feuille '{sheet_name}' chargée avec succès !")
        self.sheet_name = sheet_name


    def compute_pr(self):
        """Construit les colonnes PRD et PRF à partir de plod/plof """
        assert self.df is not None, MESSAGE_NO_DF

        prd = self.df[PLOD].str.extract(PR_REGEX)
        prf = self.df[PLOF].str.extract(PR_REGEX)

        self.df[PRD_NAT] = prd[0]
        self.df[PRD_NUM] = prd[2].astype("Int64")
        self.df[PRF_NUM] = prf[2].astype("Int64")

        # On reconstruit les labels des PR en txt
        self.df[PRD] = [
            f"{el[PRD_NAT]}{el[PRD_NUM]}"
            for _, el in self.df.iterrows()
        ]
        self.df[PRF] = self.df[PRF_NUM].apply(
            lambda x: f"PR{int(x)}" if pd.notna(x) else None
        )


    def compute_levels(self):
        """Calcul des surfaces non cumulées pour chaque état de niveau"""
        assert self.df is not None, MESSAGE_NO_DF

        for state, cols in D_SUP.items():
            for i in range(NB_LEVELS):
                try:
                    levels = self.df[cols[i]] - self.df[cols[i + 1]]
                except IndexError:
                    levels = self.df[cols[i]]
                self.df[level_name(state, i)] = levels



    def compute_percent(self):
        """Calcul des pourcentages par rapport à S_evaluee"""
        assert self.df is not None, MESSAGE_NO_DF

        for state in STATES:
            for level in range(NB_LEVELS):
                self.df[pct_name(state, level)] = (
                    self.df[level_name(state, level)] / self.df[SURF_EVAL] * 100
                )

    def set(self,route: str | None, dep: str | None) :
        """fixe route & département"""
        assert self.df is not None, MESSAGE_NO_DF
        if route :
            self.df = self.df[self.df[ROUTE] == route]
        if dep :
            self.df = self.df[self.df[DEP].astype(str).str.strip() ==
            str(dep).strip()]

    def filter(
        self,
        prd : int | None = None,
        abd : int | None = None,
        prf : int | None = None,
        abf : int | None = None

    ) -> None :
        """filtre sur pr/abs"""
        assert self.df is not None, MESSAGE_NO_DF
        if prd is not None :
            self.df = self.df[self.df[PRD_NUM] >= prd]
        if abd is not None :
            self.df = self.df[self.df[ABD] >= abd]
        if prf is not None :
            self.df = self.df[self.df[PRF_NUM] <= prf]
        if abf is not None :
            self.df = self.df[self.df[ABF] <= abf]


    def compute_curviligne(
        self,
        sens : str
    ) -> tuple [DataFrame, dict[str, float]] :
        """retourne le dataframe avec les abscisses curvilignes 
        et un dictionnaire des pr"""
        assert self.df is not None, MESSAGE_NO_DF
        # On fixe le sens
        df = self.df[self.df[SENS] == sens]
        # On trie les données dans le sens pr/abs croissants
        df = df.sort_values(by=[PRD_NUM, PRD_NAT, ABD], ascending=True)
        df[CURV_END]= df[LONGUEUR_TRONCON].cumsum()
        df[CURV_START] = df[CURV_END] - df[LONGUEUR_TRONCON]
        return df, {
            el[PRD]: el[CURV_START]
            for _, el in df.iterrows()
            if el[ABD] == 0
        }

def graphe_state_section(
    state: str,
    row : Series,
    ax: Axes
) -> None:
    """Trace un état de surface pour un tronçon donné."""
    curv_start = row[CURV_START]
    curv_end = row[CURV_END]
    # width représente la longueur totale du tronçon.
    width = curv_end - curv_start

    percents = [
        Y_SCALE * row[pct_name(state,lvl)] / 100
        for lvl in range(len(COLORS))
    ]
    bottoms  = [0, *accumulate(percents[:-1])]
    ax.bar(
        x=curv_start+width/2,
        width=width,
        bottom=bottoms,
        height=percents,
        color=COLORS
    )

class GraphStates:
    """Classe pour grapher les états de surface IQRN 3D"""
    def __init__(self, df: DataFrame | None = None):
        """initialisation"""
        if df is None:
            self.analyzer = SurfaceAnalyzer(FILE)
            self.analyzer.load_sheet()
            self.analyzer.compute_pr()
            self.analyzer.compute_levels()
            self.analyzer.compute_percent()
        else :
            self.analyzer = SurfaceAnalyzer(df=df)
        self.route : str | None = None
        self.dep : str | None = None
        self.curv_prs : dict[str, dict[str, float]] = defaultdict(dict)

    def set_route_dep(
        self,
        route: str | None,
        dep: str | None
    ) -> None:
        """fixe route & département"""
        self.route = route
        self.dep = dep
        self.analyzer.set(route=route, dep=dep)


    def graphe_sens(
        self,
        sens: str,
        axes: list[Axes],
        **kwargs
    ) -> DataFrame:
        """graphes de tous les états pour un sens donné"""
        assert self.route is not None
        assert self.dep is not None
        self.analyzer.filter(**kwargs)
        df, self.curv_prs[sens] = self.analyzer.compute_curviligne(sens)
        # Affichage des PR sur le premier axe
        for pr, curv in self.curv_prs[sens].items():
            draw_object(
                label = pr,
                x_pos = curv,
                ymax = Y_SCALE_W_PR,
                ax = axes[0],
            )
        # Décoration et habillage des graphiques
        title = f"sens {sens}"
        habille(axes[0], Y_SCALE_W_PR, title, STATES[IES], grid=True)
        habille(axes[1], Y_SCALE, title, STATES[IEP], grid=True)
        habille(axes[2], Y_SCALE, title, STATES[IETP], grid=True)

        # Valeurs
        for _, row in df.iterrows():
            graphe_state_section(IES, row, axes[0])
            graphe_state_section(IEP, row, axes[1])
            graphe_state_section(IETP, row, axes[2])
        return df
