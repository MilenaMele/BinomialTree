"""Courbes de valeur et d'exercice par date (cours, pages 22–23).

À quoi sert ce module ?
    À partir d'un arbre construit, il trace la valeur moyenne actualisée
    de l'option date par date et les probabilités d'exercice, puis les
    exporte en PNG/SVG et en CSV. L'interface l'utilise pour le bouton
    « Préparer les graphiques PNG/SVG » de la page « Valeur & exercice ».

Le mot « valeur » désigne ici une MOYENNE PAR DATE dans l'arbre, et non
une courbe du prix en fonction du spot. Toutes les valeurs sont actualisées
à la date initiale ; toutes les probabilités sont risque-neutres (mesure Q).

* ``exercise_region_probability`` : masse des nœuds où il faudrait exercer
  à cette date (courbe du PDF), même après un exercice théorique antérieur.
* ``first_exercise_probability`` : masse qui exerce POUR LA PREMIÈRE FOIS.
* ``cumulative_exercise_probability`` : cumul de la distribution précédente.

Il ne faut donc PAS additionner les valeurs de la première courbe : un même
chemin peut passer plusieurs fois par une zone d'exercice.

Exemple en ligne de commande :

    python profils_option.py --steps 200 --output-dir resultats/profils

Le moteur ``arbre_binomial.py`` fait tous les calculs financiers. Ce module
ne garde que des lignes de nombres, jamais un tableau de nœuds de l'arbre.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from math import isfinite
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from arbre_binomial import Market, Model, Option, Tree


class ProfileRow(Protocol):
    """Champs lus dans une ligne scalaire renvoyée par le moteur.

    Un ``Protocol`` sert uniquement à décrire les attributs utilisés ici :
    la même dataclass renvoyée par ``Tree.exercise_profile()`` est réutilisée.
    """

    # Mêmes champs que ExerciseProfileRow dans arbre_binomial.py.
    step: int
    time_years: float
    date: dt.date | dt.datetime
    discounted_mean_value: float
    exercise_region_probability: float
    first_exercise_probability: float
    cumulative_exercise_probability: float
    survival_probability: float
    discounted_alive_value: float
    discounted_exercise_cashflow: float
    cumulative_discounted_cashflow: float
    total_value: float


# Cet ordre est aussi celui des colonnes du CSV. Les noms sont identiques
# aux attributs du moteur pour faciliter le passage du graphique au code.
CSV_FIELDS = (
    "step", "time_years", "date", "discounted_mean_value",
    "exercise_region_probability", "first_exercise_probability",
    "cumulative_exercise_probability", "survival_probability",
    "discounted_alive_value", "discounted_exercise_cashflow",
    "cumulative_discounted_cashflow", "total_value",
)


class OptionProfiles:
    """Calcule, trace et exporte les profils d'un ``Tree``.

    ``compute()`` renvoie une ligne par date, racine et maturité incluses.
    ``plot()`` produit deux figures, chacune en PNG et en SVG, et un CSV.
    Après modification du strike ou du type d'exercice, relancer
    ``compute()`` ou ``plot()`` pour recalculer les valeurs.
    """

    def __init__(self, tree: Tree) -> None:
        if not isinstance(tree, Tree):
            raise TypeError("OptionProfiles attend un objet Tree.")
        self.tree = tree
        # Dernières lignes calculées (nombres seulement), None avant compute().
        self.results: tuple[ProfileRow, ...] | None = None

    def compute(self) -> tuple[ProfileRow, ...]:
        """Construire si nécessaire et demander un profil repricé au moteur.

        ``Tree.build()`` est idempotente et ``exercise_profile()`` refait le
        pricing : on ne lit jamais des valeurs d'option périmées.
        """
        self.tree.build()
        self.results = tuple(self.tree.exercise_profile())
        if not self.results:
            raise RuntimeError("Le moteur a renvoyé un profil vide.")
        return self.results

    def rows(self) -> tuple[ProfileRow, ...]:
        """Fournir les résultats ; lancer le calcul au premier besoin."""
        if self.results is None:
            return self.compute()
        return self.results

    # ------------------------------------------------------------------
    # Mise en forme commune des figures.
    # ------------------------------------------------------------------
    def _description(self) -> str:
        """Paramètres courts pour identifier une figure dans un rapport."""
        tree = self.tree
        exercise = ("américain" if tree.option.exercise_type.lower() == "american"
                    else "européen")
        return (
            f"{tree.option.type.capitalize()} {exercise} — "
            f"S₀={tree.market.spot:g}, K={tree.option.strike:g}, "
            f"r={tree.market.rate:.1%}, σ={tree.market.vol:.1%}, "
            f"N={tree.model.nb_steps}, pruning={tree.model.pruning_threshold:g}"
        )

    def _decorate_time_axis(self, ax: Axes) -> None:
        """Axe du temps, grille et repère de la date exacte du dividende cash."""
        tree = self.tree
        ax.set_xlabel(f"Temps depuis le {tree.model.pricing_date:%d/%m/%Y} "
                      "(années, ACT/365)")
        ax.grid(alpha=0.25)
        ex_date = tree.market.ex_dividend_date
        if not (tree.market.dividend_amount > 0 and ex_date is not None
                and tree.model.pricing_date < ex_date <= tree.option.maturity):
            return
        # Le maillage peut ne pas contenir exactement la date cash : la ligne
        # indique la vraie date ; le moteur affecte le saut au pas qui la contient.
        ex_time = (ex_date - tree.model.pricing_date).days / 365.0
        ax.axvline(ex_time, color="0.45", linestyle=":", linewidth=1.2,
                   label=f"Dividende {tree.market.dividend_amount:g} le {ex_date:%d/%m/%Y}")

    @staticmethod
    def _save_figure(fig: Figure, output_dir: Path, stem: str) -> dict[str, Path]:
        """Sauvegarder la même figure en bitmap et en format vectoriel."""
        files = {"png": output_dir / f"{stem}.png", "svg": output_dir / f"{stem}.svg"}
        fig.savefig(files["png"], dpi=180, bbox_inches="tight")
        # svg.fonttype='none' garde le texte sélectionnable dans le SVG.
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(files["svg"], bbox_inches="tight")
        return files

    # ------------------------------------------------------------------
    # Figure 1 : valeur moyenne par date et contrôle de conservation.
    # ------------------------------------------------------------------
    def _draw_mean_value(self, ax: Axes) -> None:
        """Moyenne du PDF : exp(-r t) Σ_j Q_j V_j sur TOUS les nœuds de la date."""
        rows = self.rows()
        times = [row.time_years for row in rows]
        initial_price = rows[0].total_value
        # Européenne : constante (tour des espérances). Américaine :
        # supermartingale, qui décroît lorsque l'exercice devient optimal.
        ax.plot(times, [row.discounted_mean_value for row in rows], color="tab:blue",
                linewidth=2, label="Moyenne brute actualisée (PDF, arbre entier)")
        ax.axhline(initial_price, color="black", linestyle="--", linewidth=1,
                   label=f"Prix initial : {initial_price:.6f}")
        ax.set_title("Moyenne de la valeur d'option par date")
        ax.set_ylabel("Valeur actualisée à la date initiale")
        self._decorate_time_axis(ax)
        ax.legend(fontsize=9)

    def _draw_conservation(self, ax: Axes) -> None:
        """Valeur vivante avant décision + flux STRICTEMENT antérieurs = prix."""
        rows = self.rows()
        times = [row.time_years for row in rows]
        # Le flux de la date est déjà dans la valeur vivante AVANT décision :
        # l'ajouter à nouveau le compterait deux fois.
        previous_cashflows = [row.cumulative_discounted_cashflow
                              - row.discounted_exercise_cashflow for row in rows]
        ax.plot(times, [row.discounted_alive_value for row in rows], color="tab:blue",
                label="Contrats vivants, avant décision")
        ax.plot(times, previous_cashflows, color="tab:orange",
                label="Cashflows reçus strictement avant cette date")
        ax.plot(times, [row.total_value for row in rows], color="tab:green", linewidth=2,
                linestyle="--", label="Somme : prix initial conservé")
        ax.set_title("Contrôle : valeur vivante + cashflows passés")
        ax.set_ylabel("Valeur actualisée à la date initiale")
        self._decorate_time_axis(ax)
        ax.legend(fontsize=9)

    def _value_figure(self) -> Figure:
        """Distinguer moyenne du PDF et valeur des contrats encore vivants."""
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
        fig.suptitle(self._description(), fontsize=12)
        self._draw_mean_value(axes[0])
        self._draw_conservation(axes[1])
        return fig

    # ------------------------------------------------------------------
    # Figure 2 : zone d'exercice, premier exercice et cumul.
    # ------------------------------------------------------------------
    def _draw_region(self, ax: Axes) -> None:
        """Courbe du PDF : masse des nœuds en zone d'exercice à chaque date."""
        rows = self.rows()
        # La dernière ligne est spéciale : à T il s'agit du payoff final,
        # pas d'un exercice ANTICIPÉ. On la dessine donc séparément.
        ax.plot([row.time_years for row in rows[:-1]],
                [row.exercise_region_probability for row in rows[:-1]],
                color="tab:blue", label="Zone d'exercice anticipé, arbre entier")
        ax.scatter([rows[-1].time_years], [rows[-1].exercise_region_probability],
                   marker="D", color="tab:purple", zorder=3,
                   label="À maturité : payoff > 0, arbre entier")
        ax.set_title("Courbe du PDF : masse des nœuds en zone d'exercice")
        ax.text(0.02, 0.96, "Un même chemin peut contribuer à plusieurs dates : "
                "ne pas sommer cette courbe.", transform=ax.transAxes, va="top",
                fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

    def _draw_first_exercise(self, ax: Axes) -> None:
        """Distribution du premier exercice (probabilités PAR PAS)."""
        rows = self.rows()
        maturity = rows[-1]
        # Chaque contrat est absorbé à son premier exercice : pas de double comptage.
        ax.bar([row.time_years for row in rows[:-1]],
               [row.first_exercise_probability for row in rows[:-1]],
               width=self.tree.deltaT * 0.75, color="tab:orange",
               label="Premier exercice anticipé à cette date")
        ax.vlines(maturity.time_years, 0, maturity.first_exercise_probability,
                  color="tab:purple", linewidth=2)
        ax.scatter([maturity.time_years], [maturity.first_exercise_probability],
                   marker="D", color="tab:purple", zorder=3,
                   label="Premier exercice à maturité, payoff > 0")
        ax.set_title("Distribution du premier exercice : aucun double comptage")

    def _draw_cumulative(self, ax: Axes) -> None:
        """Probabilité cumulée d'avoir exercé et survie avant T."""
        rows = self.rows()
        maturity = rows[-1]
        ax.step([row.time_years for row in rows],
                [row.cumulative_exercise_probability for row in rows],
                where="post", color="tab:green", label="Probabilité cumulée d'avoir exercé")
        ax.step([row.time_years for row in rows[:-1]],
                [row.survival_probability for row in rows[:-1]], where="post",
                color="tab:red", linestyle="--", label="Contrat vivant après décision, avant T")
        # À T, survival_probability est la masse qui expire sans paiement :
        # on la distingue par une croix à la date T.
        ax.scatter([maturity.time_years], [maturity.survival_probability], color="tab:red",
                   marker="x", s=55, zorder=3, label="À T : expiration sans paiement")
        ax.set_title("Cumul et survie — expiration sans exercice à T : "
                     f"{maturity.survival_probability:.2%}")

    def _format_probability_axes(self, axes: list[Axes]) -> None:
        """Axes en pourcentage ; plage adaptée pour les probabilités par pas."""
        maximum = max(row.first_exercise_probability for row in self.rows())
        for index, ax in enumerate(axes):
            ax.set_ylabel("Probabilité risque-neutre")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            # Une probabilité PAR PAS est souvent de quelques % : une échelle
            # de 0 à 100 % écraserait la distribution près de zéro.
            if index == 1:
                ax.set_ylim(0.0, min(1.08, max(0.01, 1.2 * maximum)))
            else:
                ax.set_ylim(-0.03, 1.08)
            self._decorate_time_axis(ax)
            ax.legend(fontsize=9, loc="upper left" if index == 2 else "best")

    def _exercise_figure(self) -> Figure:
        """Dessiner la zone brute, la première distribution, puis son cumul."""
        fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
        fig.suptitle(self._description() + "\nProbabilités risque-neutres Q", fontsize=12)
        self._draw_region(axes[0])
        self._draw_first_exercise(axes[1])
        self._draw_cumulative(axes[2])
        self._format_probability_axes(list(axes))
        return fig

    # ------------------------------------------------------------------
    # Exports CSV / PNG / SVG.
    # ------------------------------------------------------------------
    def export_csv(self, output_dir: str | Path) -> Path:
        """Exporter les nombres en précision complète, avec une date ISO.

        Les probabilités sont entre 0 et 1 (0.25 signifie 25 %), sans
        actualisation. Les champs ``discounted_*`` sont déjà actualisés.
        """
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = destination / "profils_option.csv"
        # Séparateur virgule ; une ligne par date de l'arbre.
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in self.rows():
                values = {field: getattr(row, field) for field in CSV_FIELDS}
                values["date"] = row.date.isoformat()
                writer.writerow(values)
        return csv_path

    def plot(self, output_dir: str | Path, show: bool = False) -> dict[str, Path]:
        """Recalculer puis écrire les cinq fichiers et renvoyer leurs chemins.

        ``show=False`` convient aux scripts, aux serveurs et à Streamlit.
        """
        self.compute()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        # Les deux figures sont créées avant plt.show() pour les afficher
        # en même temps. finally ferme uniquement nos propres figures.
        value_figure = self._value_figure()
        exercise_figure = self._exercise_figure()
        try:
            value = self._save_figure(value_figure, destination, "valeur_option_par_date")
            exercise = self._save_figure(exercise_figure, destination,
                                         "probabilite_exercice_par_date")
            csv_path = self.export_csv(destination)
            if show:
                plt.show()
        finally:
            plt.close(value_figure)
            plt.close(exercise_figure)
        return {"value_png": value["png"], "value_svg": value["svg"],
                "exercise_png": exercise["png"], "exercise_svg": exercise["svg"],
                "csv": csv_path}


# -----------------------------------------------------------------------------
# Lancement autonome (ligne de commande), sans Streamlit.
# -----------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    """Validation argparse, avec une erreur lisible plutôt qu'une traceback."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entier strictement positif attendu") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("le nombre de pas doit être >= 1")
    return value


def _threshold(text: str) -> float:
    """Seuil de pruning : refuse aussi NaN, l'infini et les valeurs hors [0, 1]."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("nombre positif ou nul attendu") from exc
    if not isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("le seuil doit être fini et compris entre 0 et 1")
    return value


def _parse_arguments() -> argparse.Namespace:
    """Options de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=_positive_int, default=200,
                        help="nombre de pas de l'arbre (défaut : 200)")
    parser.add_argument("--pruning-threshold", type=_threshold, default=1e-7,
                        help="seuil de pruning ; 0 le désactive (défaut : 1e-7)")
    parser.add_argument("--exercise-type", choices=("american", "european"),
                        default="american", help="style d'exercice (défaut : american)")
    parser.add_argument("--output-dir", type=Path, default=Path("resultats/profils"),
                        help="dossier de sortie (défaut : resultats/profils)")
    parser.add_argument("--show", action="store_true", help="afficher les figures")
    return parser.parse_args()


def main() -> None:
    """Démonstration reproductible : put américain avec dividende cash."""
    args = _parse_arguments()
    tree = Tree(
        model=Model(pricing_date=dt.date(2026, 9, 15), nb_steps=args.steps,
                    pruning_threshold=args.pruning_threshold),
        option=Option(strike=101.0, maturity=dt.date(2027, 9, 2),
                      type="Put", exercise_type=args.exercise_type),
        market=Market(spot=100.0, rate=0.07, vol=0.30, dividend_yield=0.0,
                      dividend_amount=2.0, ex_dividend_date=dt.date(2027, 3, 15)),
    )
    study = OptionProfiles(tree)
    files = study.plot(args.output_dir, show=args.show)
    rows = study.rows()
    # Résumé dans le terminal.
    print(f"Prix initial : {rows[0].total_value:.10f}")
    print(f"Exercice cumulé : {rows[-1].cumulative_exercise_probability:.4%}")
    for name, path in files.items():
        print(f"  {name} : {path.resolve()}")


if __name__ == "__main__":
    main()
