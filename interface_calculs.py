# =============================================================================
# Couche de calcul de l'interface, indépendante de Streamlit.
#
# Il fait le lien entre les champs de l'écran et le moteur
# ``arbre_binomial.py`` : il valide les paramètres, construit les arbres,
# lance les études (convergence, pruning, dividendes, taux, comparaison
# des exercices) et renvoie des tableaux pandas prêts à afficher. Il
# prépare aussi les grecques (module grecques.py) et le P&L de couverture.
#
# =============================================================================

# Imports : bibliothèque standard, paquets externes puis modules du projet.
from __future__ import annotations

import datetime as dt
import gc
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from functools import partial
from io import StringIO
from math import exp, isfinite
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, Literal

# Paquet externe (tableaux de résultats).
import pandas as pd

# Modules du projet : moteur, Black-Scholes et débogueur.
from arbre_binomial import Market, Model, Option, Tree
from convergence import Convergence
from debug_arbre import DebugTree
from grecques import (GREEK_NAMES, TreeGreeks, black_scholes_greeks, taylor_pnl,
                      tree_node_delta_gamma)


# Ces limites protègent un lancement interactif contre un calcul involontaire
# de plusieurs millions de nœuds par scénario. Elles ne changent pas le moteur.
MAX_STEPS = 2_000
MAX_STUDY_POINTS = 20
MAX_STRIKES = 81
MAX_DEBUG_STEPS = 40
# Les quatre contrats comparés sur un même arbre (clé = type_exercice).
CONTRACT_KEYS = ("call_european", "call_american", "put_european", "put_american")
# Scénarios du graphique « probabilité d'exercice par date » (cours p. 23).
EXERCISE_SCENARIOS = {
    "american_dividends": "Américaine, dividendes du scénario",
    "american_no_dividend": "Américaine sans dividende (même forward)",
    "european_no_dividend": "Européenne sans dividende (même forward)",
    "european_dividends": "Européenne, dividendes du scénario",
}


# -----------------------------------------------------------------------------
# Validation des valeurs saisies.
# -----------------------------------------------------------------------------
def _finite_number(name: str, value: float) -> float:
    """Refuser aussi bool, NaN et infini, puis rendre un float Python."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} doit être un nombre réel fini.")
    return float(value)


# Une grille comporte un entier de pas, sans arrondir 20,5 vers 20
def _validate_steps(value: int, maximum: int = MAX_STEPS) -> int:
    """Une grille comporte un entier de pas, sans arrondir 20,5 vers 20."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"Le nombre de pas doit être un entier entre 1 et {maximum}.")
    return int(value)


# Seuil de pruning fini, compris entre 0 et 1
def _validate_threshold(value: float) -> float:
    """Seuil de pruning fini, compris entre 0 et 1."""
    value = _finite_number("Le seuil de pruning", value)
    if not 0 <= value <= 1:
        raise ValueError("Le seuil de pruning doit être compris entre 0 et 1.")
    return value


# datetime hérite de date en Python, mais le moteur demande une date seule
def _validate_date(name: str, value: dt.date) -> None:
    """datetime hérite de date en Python, mais le moteur demande une date seule."""
    if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
        raise ValueError(f"{name} doit être une date sans heure (datetime.date).")


# Photographie immuable des paramètres saisis dans le formulaire
@dataclass(frozen=True)
class Parameters:
    """Photographie immuable des paramètres saisis dans le formulaire.

    Les taux sont des décimaux : 0.03 signifie 3 %, et non 0,03 %.
    Le rendement continu ``dividend_yield`` et le cash ``dividend_amount``
    peuvent coexister ; la date du cash est obligatoire s'il est positif.
    ``frozen=True`` interdit de modifier un résultat déjà calculé : pour
    changer un champ, ``replace(params, rate=0.0)`` crée une nouvelle copie.
    """

    # Dates du contrat.
    pricing_date: dt.date
    maturity: dt.date
    # Marché.
    spot: float
    strike: float
    rate: float
    vol: float
    dividend_yield: float
    dividend_amount: float
    ex_dividend_date: dt.date | None
    # Maillage et contrat.
    nb_steps: int
    pruning_threshold: float
    option_type: Literal["Call", "Put"]
    exercise_type: Literal["European", "American"]

    # Nombres finis, spot positif, autres grandeurs positives ou nulles
    def _validate_market(self) -> None:
        """Nombres finis, spot positif, autres grandeurs positives ou nulles."""
        for name in ("spot", "strike", "rate", "vol", "dividend_yield", "dividend_amount"):
            _finite_number(name, getattr(self, name))
        if self.spot <= 0:
            raise ValueError("Le spot doit être strictement positif.")
        if min(self.strike, self.vol, self.dividend_yield, self.dividend_amount) < 0:
            raise ValueError("Strike, volatilité, rendement continu et cash doivent "
                             "être positifs ou nuls.")
        if self.ex_dividend_date is not None:
            _validate_date("La date de détachement", self.ex_dividend_date)
        if self.dividend_amount > 0 and self.ex_dividend_date is None:
            raise ValueError("Un dividende cash positif nécessite sa date de détachement.")

    # Vérifier tous les champs avant de commencer un calcul coûteux
    def validate(self) -> None:
        """Vérifier tous les champs avant de commencer un calcul coûteux."""
        _validate_date("La date de calcul", self.pricing_date)
        _validate_date("La maturité", self.maturity)
        if self.maturity <= self.pricing_date:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        self._validate_market()
        _validate_steps(self.nb_steps)
        _validate_threshold(self.pruning_threshold)
        # Libellés exacts attendus par le moteur.
        if self.option_type not in ("Call", "Put"):
            raise ValueError("Le type d'option doit être 'Call' ou 'Put'.")
        if self.exercise_type not in ("European", "American"):
            raise ValueError("Le type d'exercice doit être 'European' ou 'American'.")


# -----------------------------------------------------------------------------
# Construction et pricing.
# -----------------------------------------------------------------------------
def has_active_cash(params: Parameters) -> bool:
    """Un cash à la date initiale est déjà détaché ; un cash après T est futur."""
    return bool(
        params.dividend_amount > 0
        and params.ex_dividend_date is not None
        and params.pricing_date < params.ex_dividend_date <= params.maturity
    )


# Traduire les paramètres de l'écran dans l'API du moteur et construire
def make_tree(params: Parameters, nb_steps: int | None = None,
              pruning_threshold: float | None = None) -> Tree:
    """Traduire les paramètres de l'écran dans l'API du moteur et construire.

    ``nb_steps`` et ``pruning_threshold`` sont des substitutions explicites :
    les études peuvent ainsi isoler N, le pruning ou la taille du SVG sans
    changer les paramètres de la valorisation principale.
    """
    params.validate()
    steps = _validate_steps(params.nb_steps if nb_steps is None else nb_steps)
    threshold = _validate_threshold(
        params.pruning_threshold if pruning_threshold is None else pruning_threshold)
    market, option, model = _engine_objects(params, steps, threshold)
    return Tree(model=model, option=option, market=market).build()


# Les trois objets du moteur (Market, Option, Model) à partir de l'écran
def _engine_objects(params: Parameters, nb_steps: int,
                    threshold: float) -> tuple[Market, Option, Model]:
    """Traduire les paramètres validés en objets du moteur, sans construire."""
    market = Market(spot=params.spot, rate=params.rate, vol=params.vol,
                    dividend_yield=params.dividend_yield,
                    dividend_amount=params.dividend_amount,
                    ex_dividend_date=params.ex_dividend_date)
    option = Option(params.strike, params.maturity, params.option_type,
                    params.exercise_type)
    return market, option, Model(params.pricing_date, nb_steps, threshold)


# Ne jamais afficher un prix NaN/inf comme s'il était un résultat valide
def _price(tree: Tree) -> float:
    """Ne jamais afficher un prix NaN/inf comme s'il était un résultat valide."""
    value = float(tree.price())
    if not isfinite(value):
        raise ValueError("Le calcul produit un prix non fini ; réduire les paramètres extrêmes.")
    return value


# Revaloriser quatre contrats sur UN seul arbre du même sous-jacent
def _four_prices(tree: Tree) -> dict[str, float]:
    """Revaloriser quatre contrats sur UN seul arbre du même sous-jacent.

    Strike, call/put et exercice ne changent pas la construction de l'arbre.
    On remet l'objet Option d'origine à la fin ; le demandeur qui veut son
    profil appelle ensuite exercise_profile(), qui le revalorise.
    """
    original = tree.option
    result: dict[str, float] = {}
    try:
        for kind in ("Call", "Put"):
            for exercise in ("European", "American"):
                tree.option = Option(original.strike, original.maturity, kind, exercise)
                result[f"{kind.lower()}_{exercise.lower()}"] = _price(tree)
    finally:
        tree.option = original
    return result


# Construire la référence européenne seulement lorsqu'elle est exacte
def _black_scholes(params: Parameters) -> Convergence:
    """Construire la référence européenne seulement lorsqu'elle est exacte.

    Un montant cash dont la date est hors contrat est sans effet : on le
    retire seulement de la référence. La référence garde le rendement q.
    """
    if has_active_cash(params):
        raise ValueError("Black-Scholes n'est pas une référence exacte avec un dividende "
                         "cash actif. Désactiver le cash pour cette étude.")
    return Convergence(
        Market(params.spot, params.rate, params.vol, params.dividend_yield),
        Option(params.strike, params.maturity, params.option_type, "European"),
        params.pricing_date,
    )


# -----------------------------------------------------------------------------
# Synthèse du scénario principal (pages « Prix » et « Valeur & exercice »).
# -----------------------------------------------------------------------------
def _black_reference(params: Parameters) -> tuple[float | None, str]:
    """Prix Black-Scholes s'il est une référence exacte, et la raison sinon."""
    if has_active_cash(params):
        return None, "Pas de référence Black-Scholes exacte avec un dividende cash actif."
    if params.exercise_type == "American":
        return None, ("Black-Scholes valorise le contrat européen ; "
                      "le contrat choisi est américain.")
    price = float(_black_scholes(params).black_price(params.strike))
    return price, "Référence européenne exacte avec rendement continu q, sans cash actif."


# Tableau des quatre prix call/put × européen/américain
def _comparison_table(four: dict[str, float]) -> pd.DataFrame:
    """Tableau des quatre prix call/put × européen/américain."""
    return pd.DataFrame([
        {"option_type": kind, "exercise_type": exercise,
         "price": four[f"{kind.lower()}_{exercise.lower()}"]}
        for kind in ("Call", "Put") for exercise in ("European", "American")
    ])


# Contrôles de masse/valeur et probabilités résumées du profil
def _profile_statistics(rows: tuple, price: float) -> dict:
    """Contrôles de masse/valeur et probabilités résumées du profil."""
    terminal = rows[-1]
    return {
        # max|survie + exercice cumulé − 1| et max|valeur conservée − prix|.
        "mass_error": max(abs(row.survival_probability
                              + row.cumulative_exercise_probability - 1) for row in rows),
        "value_error": max(abs(row.total_value - price) for row in rows),
        "early_exercise_probability": sum(row.first_exercise_probability
                                          for row in rows[:-1]),
        "maturity_exercise_probability": terminal.first_exercise_probability,
        "expiration_probability": terminal.survival_probability,
    }


# Prix, comparaison et profil du contrat choisi, à partir d'un arbre
def _summary_of_tree(tree: Tree, params: Parameters) -> dict:
    """Prix, comparaison et profil du contrat choisi, à partir d'un arbre."""
    four = _four_prices(tree)
    # Le profil reprice le contrat INITIAL (et non le dernier put comparé).
    rows = tree.exercise_profile()
    price = float(tree.root.option_value)
    if not isfinite(price):
        raise ValueError("Le calcul produit un prix non fini.")
    black_price, black_reason = _black_reference(params)
    summary = {
        "price": price,
        "comparison": _comparison_table(four),
        "node_count": tree.count_nodes(),
        "black_scholes": black_price,
        "black_scholes_reason": black_reason,
        # asdict ne porte que sur les lignes de résultats, jamais sur un nœud.
        "profile": pd.DataFrame([asdict(row) for row in rows]),
    }
    summary.update(_profile_statistics(rows, price))
    return summary


# Prix, comparaisons et profils correspondant au contrat choisi à l'écran
def calculate_summary(params: Parameters) -> dict:
    """Prix, comparaisons et profils correspondant au contrat choisi à l'écran.

    L'arbre est construit une seule fois. ``mass_error`` et ``value_error``
    contrôlent le modèle effectivement calculé, élagué ou non.
    """
    params.validate()
    gc.collect()
    start = perf_counter()
    tree = None
    try:
        tree = make_tree(params)
        summary = _summary_of_tree(tree, params)
        # Temps de la synthèse entière : construction, quatre prix, profil.
        summary["elapsed_seconds"] = perf_counter() - start
        return summary
    finally:
        del tree
        gc.collect()


# -----------------------------------------------------------------------------
# Comparaison des probabilités d'exercice par date (cours p. 23).
# -----------------------------------------------------------------------------
def _same_forward_without_dividend(params: Parameters, tree: Tree) -> Parameters:
    """Scénario sans dividende dont le forward à maturité est identique.

    Comme dans le cours, on ajuste le spot : S0' = F(T) × exp(−rT), où F(T)
    est le forward du scénario avec dividendes (cash et rendement q).
    """
    maturity_years = tree.duration_days / 365.0
    forward = tree.mean_at_time(maturity_years)
    adjusted_spot = forward * exp(-params.rate * maturity_years)
    return replace(params, spot=adjusted_spot, dividend_yield=0.0,
                   dividend_amount=0.0, ex_dividend_date=None)


# Profil d'un contrat (même arbre, exercice imposé) au format long
def _profile_frame(tree: Tree, exercise: str, label: str) -> pd.DataFrame:
    """Profil d'un contrat (même arbre, exercice imposé) au format long."""
    original = tree.option
    try:
        tree.option = Option(original.strike, original.maturity, original.type, exercise)
        rows = tree.exercise_profile()
    finally:
        tree.option = original
    # On garde seulement les colonnes utiles aux deux graphiques de la page 23.
    return pd.DataFrame({
        "scenario": label,
        "step": [row.step for row in rows],
        "time_years": [row.time_years for row in rows],
        "exercise_region_probability": [row.exercise_region_probability for row in rows],
        "discounted_mean_value": [row.discounted_mean_value for row in rows],
        "price": rows[0].total_value,
    })


# Profils des scénarios demandés partageant le même arbre
def _scenario_profiles(params: Parameters, keys: list[str],
                       suffix: str) -> list[pd.DataFrame]:
    """Profils des scénarios demandés partageant le même arbre."""
    frames = []
    tree = None
    try:
        tree = make_tree(params)
        for exercise in ("American", "European"):
            key = f"{exercise.lower()}_{suffix}"
            if key in keys:
                frames.append(_profile_frame(tree, exercise, EXERCISE_SCENARIOS[key]))
        return frames
    finally:
        del tree
        gc.collect()


# Probabilité d'exercice et valeur moyenne par date pour plusieurs contrats
def exercise_comparison(params: Parameters, keys: list[str]) -> pd.DataFrame:
    """Probabilité d'exercice et valeur moyenne par date pour plusieurs contrats.

    Pour chaque date, ``exercise_region_probability`` est la somme des
    probabilités des nœuds où l'exercice est décidé (définition du cours).
    Les contrats « sans dividende » gardent le même forward à maturité.
    """
    params.validate()
    if not keys or any(key not in EXERCISE_SCENARIOS for key in keys):
        raise ValueError("Choisir au moins un scénario de la liste.")
    reference = make_tree(params)
    no_dividend = _same_forward_without_dividend(params, reference)
    # Les liens entre nœuds forment des cycles : on libère l'arbre explicitement.
    del reference
    gc.collect()
    # Un arbre avec dividendes et un arbre sans dividende (même forward).
    frames = (_scenario_profiles(params, keys, "dividends")
              + _scenario_profiles(no_dividend, keys, "no_dividend"))
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Études répétant plusieurs calculs (convergence, strike, pruning...).
# -----------------------------------------------------------------------------
def _study_values(name: str, values: list, maximum: int = MAX_STUDY_POINTS) -> list:
    """Valider toute la série avant de commencer la première construction."""
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} doit contenir entre 1 et {maximum} valeurs.")
    return list(values)


# Liste de nombres de pas, chacun validé
def _step_values(steps: list[int]) -> list[int]:
    """Liste de nombres de pas, chacun validé."""
    return [_validate_steps(value)
            for value in _study_values("La liste des nombres de pas", steps)]


# Remplir une ligne d'étude ; un échec est noté au lieu d'arrêter l'étude
def _guarded_row(row: dict, compute: Callable[[], dict]) -> dict:
    """Remplir une ligne d'étude ; un échec est noté au lieu d'arrêter l'étude."""
    start = perf_counter()
    gc.collect()
    try:
        row.update(compute())
    except (ValueError, ArithmeticError) as exc:
        row.update(status="error", error_message=str(exc))
    finally:
        row["seconds"] = perf_counter() - start
        gc.collect()
    return row


# Prix à N pas et écart à Black-Scholes (brut et multiplié par N)
def _convergence_point(params: Parameters, n: int, reference: float) -> dict:
    """Prix à N pas et écart à Black-Scholes (brut et multiplié par N)."""
    tree = make_tree(params, nb_steps=n)
    value = _price(tree)
    return {"price": value, "error": value - reference,
            "scaled_error": (value - reference) * n, "node_count": tree.count_nodes()}


# Isoler l'erreur de discrétisation : contrat EUROPÉEN, pruning nul
def convergence_study(params: Parameters, steps: list[int]) -> pd.DataFrame:
    """Isoler l'erreur de discrétisation : contrat EUROPÉEN, pruning nul.

    Une erreur de maillage est conservée dans la ligne correspondante au lieu
    de faire disparaître tous les autres résultats de l'étude.
    """
    params.validate()
    numbers = _step_values(steps)
    reference = float(_black_scholes(params).black_price(params.strike))
    european = replace(params, exercise_type="European", pruning_threshold=0.0)
    results = []
    # Une ligne par N, avec des valeurs NaN tant que le calcul n'a pas réussi.
    for n in numbers:
        row = {"nb_steps": n, "price": float("nan"), "reference": reference,
               "error": float("nan"), "scaled_error": float("nan"),
               "seconds": 0.0, "node_count": 0, "status": "ok", "error_message": ""}
        results.append(_guarded_row(row, partial(_convergence_point, european, n, reference)))
    return pd.DataFrame(results)


# Un seul arbre pour N : on change seulement le strike et on reprice
def _strike_rows(params: Parameters, n: int, strikes: list[float],
                 black: Convergence) -> list[dict]:
    """Un seul arbre pour N : on change seulement le strike et on reprice."""
    tree = None
    try:
        tree = make_tree(params, nb_steps=n)
        rows = []
        for strike in strikes:
            tree.option.strike = strike
            value = _price(tree)
            reference = float(black.black_price(strike))
            rows.append({"nb_steps": n, "strike": strike, "price": value,
                         "reference": reference, "error": value - reference})
        return rows
    finally:
        del tree
        gc.collect()


# Tracer l'erreur selon K ; ne construire qu'un seul arbre pour chaque N
def strike_study(params: Parameters, strikes: list[float], steps: list[int]) -> pd.DataFrame:
    """Tracer l'erreur selon K ; ne construire qu'un seul arbre pour chaque N.

    Comme la convergence en N, cette comparaison utilise le contrat européen
    sans pruning et refuse un cash actif.
    """
    params.validate()
    numbers = _step_values(steps)
    strike_values = [_finite_number("Le strike de l'étude", k)
                     for k in _study_values("La liste des strikes", strikes, MAX_STRIKES)]
    if any(k < 0 for k in strike_values):
        raise ValueError("Les strikes doivent être positifs ou nuls.")
    black = _black_scholes(params)
    european = replace(params, exercise_type="European", pruning_threshold=0.0)
    results = []
    # Un arbre par N, réutilisé pour tous les strikes.
    for n in numbers:
        gc.collect()
        results.extend(_strike_rows(european, n, strike_values, black))
    return pd.DataFrame(results)


# Compter les transitions coupées par liens directs, sans liste de nœuds
def _pruned_count(tree: Tree) -> int:
    """Compter les transitions coupées par liens directs, sans liste de nœuds."""
    return sum(int(node.is_pruned)
               for trunk in tree.iter_columns() for node in tree.iter_column(trunk))


# Prix, taille et nombre de coupes de l'arbre pour un seuil donné
def _pruning_point(params: Parameters, n: int, threshold: float) -> dict:
    """Prix, taille et nombre de coupes de l'arbre pour un seuil donné."""
    tree = make_tree(params, nb_steps=n, pruning_threshold=threshold)
    return {"price": _price(tree), "node_count": tree.count_nodes(),
            "pruned_count": _pruned_count(tree)}


# Comparer les seuils au même arbre sans pruning, au N demandé
def pruning_study(params: Parameters, thresholds: list[float], nb_steps: int) -> pd.DataFrame:
    """Comparer les seuils au même arbre sans pruning, au N demandé.

    La référence seuil 0 est toujours calculée. Si elle échoue (possible avec
    un cash), les écarts restent NaN : on n'invente pas de référence.
    """
    params.validate()
    n = _validate_steps(nb_steps)
    values = [_validate_threshold(value)
              for value in _study_values("La liste des seuils", thresholds)]
    # Dédoublonner les seuils économise des constructions identiques. La table
    # conserve l'ordre saisi et commence par la référence explicite seuil 0.
    results = []
    reference = None
    for threshold in dict.fromkeys([0.0, *values]):
        row = {"threshold": threshold, "price": float("nan"), "price_gap": float("nan"),
               "node_count": 0, "seconds": 0.0, "pruned_count": 0,
               "status": "ok", "error_message": ""}
        row = _guarded_row(row, partial(_pruning_point, params, n, threshold))
        if threshold == 0 and row["status"] == "ok":
            reference = row["price"]
        if reference is not None and row["status"] == "ok":
            row["price_gap"] = row["price"] - reference
        results.append(row)
    return pd.DataFrame(results)


# Quatre prix pour un marché modifié, et primes américaines si étude de taux
def _four_contract_point(params: Parameters, field: str, value: float) -> dict:
    """Quatre prix pour un marché modifié, et primes américaines si étude de taux."""
    tree = None
    try:
        tree = make_tree(replace(params, **{field: value}))
        prices = _four_prices(tree)
        if field == "rate":
            prices.update(call_premium=prices["call_american"] - prices["call_european"],
                          put_premium=prices["put_american"] - prices["put_european"])
        return prices
    finally:
        del tree
        gc.collect()


# Boucle commune : un arbre par marché, puis quatre rétropropagations
def _four_contract_study(params: Parameters, field: str, values: list[float]) -> pd.DataFrame:
    """Boucle commune : un arbre par marché, puis quatre rétropropagations.

    Les autres scénarios continuent si un maillage cash n'est pas admissible.
    """
    results = []
    for value in values:
        row = {field: value, **{key: float("nan") for key in CONTRACT_KEYS},
               "status": "ok", "error_message": ""}
        if field == "rate":
            row.update(call_premium=float("nan"), put_premium=float("nan"))
        row = _guarded_row(row, partial(_four_contract_point, params, field, value))
        # Le temps n'est pas affiché dans ces études.
        row.pop("seconds")
        results.append(row)
    return pd.DataFrame(results)


# Changer seulement le cash ; garder le taux et q de l'écran
def dividend_study(params: Parameters, amounts: list[float]) -> pd.DataFrame:
    """Changer seulement le cash ; garder le taux et q de l'écran."""
    params.validate()
    values = [_finite_number("Le dividende cash", value)
              for value in _study_values("La liste des dividendes", amounts)]
    if any(value < 0 for value in values):
        raise ValueError("Les dividendes cash doivent être positifs ou nuls.")
    if any(value > 0 for value in values) and params.ex_dividend_date is None:
        raise ValueError("Choisir une date de détachement pour étudier des dividendes "
                         "cash positifs.")
    return _four_contract_study(params, "dividend_amount", values)


# Comparer taux positifs/négatifs et prime américain moins européen
def rate_study(params: Parameters, rates: list[float]) -> pd.DataFrame:
    """Comparer taux positifs/négatifs et prime américain moins européen."""
    params.validate()
    values = [_finite_number("Le taux", value)
              for value in _study_values("La liste des taux", rates)]
    return _four_contract_study(params, "rate", values)


# -----------------------------------------------------------------------------
# Débogueur : contrôles et dessin SVG (page « Arbre & diagnostics »).
# -----------------------------------------------------------------------------
def _diagnostic(debugger: DebugTree, errors: int, report: str) -> dict:
    """Résumé numérique d'un contrôle du débogueur."""
    return {
        "report": report,
        "errors": int(errors),
        "node_count": int(debugger.node_count),
        "pruned_count": int(debugger.pruned_node_count),
        "weighted_mean_bias": float(debugger.pruning_weighted_mean_bias),
        "terminal_mean_bias": float(debugger.discounted_terminal_mean_bias),
    }


# Contrôler puis dessiner ; les messages imprimés deviennent le rapport
def _checked_svg(debugger: DebugTree, color_by: str, show_neighbors: bool) -> dict:
    """Contrôler puis dessiner ; les messages imprimés deviennent le rapport."""
    log = StringIO()
    # Dossier temporaire propre à cet appel : aucun fichier partagé n'est écrasé.
    with redirect_stdout(log), TemporaryDirectory(prefix="arbre_streamlit_") as folder:
        errors = debugger.check()
        svg = ""
        # Un graphe invalide ne doit pas être présenté comme un dessin vérifié.
        if not errors:
            path = debugger.export_svg(Path(folder) / "arbre.svg", color_by,
                                       show_neighbors, run_checks=False)
            svg = path.read_text(encoding="utf-8")
    result = _diagnostic(debugger, errors, log.getvalue())
    result["svg"] = svg
    return result


# Construire un petit arbre indépendant, vérifier puis rendre son SVG
def debug_tree(params: Parameters, steps: int, color_by: str, show_neighbors: bool) -> dict:
    """Construire un petit arbre indépendant, vérifier puis rendre son SVG.

    Le petit N est un paramètre de visualisation : il ne remplace pas le N du
    prix principal. Le graphe dessiné peut donc différer du grand arbre.
    """
    params.validate()
    n = _validate_steps(steps, MAX_DEBUG_STEPS)
    DebugTree.validate_color(color_by)
    if not isinstance(show_neighbors, bool):
        raise ValueError("show_neighbors doit être un booléen.")
    tree = None
    gc.collect()
    # Construire, pricer, puis contrôler et dessiner.
    try:
        tree = make_tree(params, nb_steps=n)
        _price(tree)
        return _checked_svg(DebugTree(tree), color_by, show_neighbors)
    finally:
        del tree
        gc.collect()


# Contrôler l'arbre au N principal sans produire un SVG gigantesque
def check_tree(params: Parameters) -> dict:
    """Contrôler l'arbre au N principal sans produire un SVG gigantesque.

    Les messages ne sont pas envoyés au terminal : seul l'écran qui a
    demandé ce contrôle les affiche.
    """
    params.validate()
    tree = None
    gc.collect()
    try:
        tree = make_tree(params)
        _price(tree)
        # Les messages imprimés par check() deviennent le rapport affiché.
        debugger = DebugTree(tree)
        log = StringIO()
        with redirect_stdout(log):
            errors = debugger.check()
        return _diagnostic(debugger, errors, log.getvalue())
    finally:
        del tree
        gc.collect()


# -----------------------------------------------------------------------------
# Grecques et couverture (page « Grecques & couverture »).
# -----------------------------------------------------------------------------
# Unité affichée de chaque grecque (vol en points, theta par jour)
GREEK_UNITS = {
    "delta": "€ par € de spot", "gamma": "delta par € de spot",
    "vega": "€ pour +1 point de vol", "vomma": "vega pour +1 point de vol",
    "vanna": "delta pour +1 point de vol", "theta": "€ par jour",
}


# Objet de calcul des grecques de l'arbre pour le scénario de l'écran
def _tree_greeks(params: Parameters, vol_shift: float) -> TreeGreeks:
    """TreeGreeks avec le N et le pruning du scénario principal."""
    params.validate()
    market, option, model = _engine_objects(params, params.nb_steps,
                                            params.pruning_threshold)
    return TreeGreeks(market, option, model, vol_shift)


# Grecques de Black-Scholes si elles sont une référence exacte, sinon None
def _black_greeks(params: Parameters) -> tuple[dict[str, float] | None, str]:
    """Référence exacte seulement pour une européenne sans cash actif."""
    if params.exercise_type == "American":
        return None, "Pas de grecques Black-Scholes exactes pour une américaine."
    if has_active_cash(params):
        return None, "Pas de grecques Black-Scholes exactes avec un dividende cash actif."
    if params.vol <= 0 or params.strike <= 0:
        return None, "Black-Scholes demande une volatilité et un strike positifs."
    market, option, _ = _engine_objects(params, 1, 0.0)
    return (black_scholes_greeks(market, option, params.pricing_date),
            "Référence exacte : Black-Scholes avec rendement continu q.")


# Delta et gamma lus sur les trois nœuds du pas 2 (astuce du cours)
def _node_delta_gamma(params: Parameters) -> tuple[float, float]:
    """Un seul arbre : delta et gamma déduits des nœuds voisins du spot."""
    tree = None
    try:
        tree = make_tree(params)
        return tree_node_delta_gamma(tree)
    finally:
        del tree
        gc.collect()


# Américaine : correction par variable de contrôle (européenne Black-Scholes)
def _control_variate(params: Parameters, values: dict[str, float],
                     vol_shift: float) -> dict[str, float] | None:
    """Grecque américaine + (Black-Scholes européenne − arbre européen).

    L'erreur de discrétisation de l'arbre varie avec σ et biaise surtout le
    vomma ; elle est presque la même pour l'américaine et l'européenne du
    même arbre. On la mesure sur l'européenne, où Black-Scholes est exact.
    """
    european = replace(params, exercise_type="European")
    black, _ = _black_greeks(european)
    if params.exercise_type != "American" or black is None:
        return None
    tree_european = _tree_greeks(european, vol_shift).all()
    return {name: values[name] + black[name] - tree_european[name] for name in GREEK_NAMES}


# Tableau comparatif : arbre, Black-Scholes et écart pour chaque grecque
def _greek_table(values: dict[str, float], black: dict[str, float] | None,
                 nodes: tuple[float, float],
                 corrected: dict[str, float] | None) -> pd.DataFrame:
    """Une ligne par grecque, avec l'unité et la méthode des nœuds."""
    rows = []
    for name in GREEK_NAMES:
        reference = None if black is None else black[name]
        node_value = {"delta": nodes[0], "gamma": nodes[1]}.get(name)
        rows.append({"greek": name, "tree": values[name], "tree_nodes": node_value,
                     "tree_control_variate": None if corrected is None else corrected[name],
                     "black_scholes": reference,
                     "gap": None if reference is None else values[name] - reference,
                     "unit": GREEK_UNITS[name]})
    return pd.DataFrame(rows)


# Prix, six grecques, comparaison Black-Scholes et méthode des nœuds
def greeks_summary(params: Parameters, vol_shift: float = 0.01) -> dict:
    """Toutes les grecques du scénario, en nombres et en tableau.

    Chaque grecque de l'arbre vient de différences finies centrées
    (``OneDimDerivative``) ; le theta est la variation sur un jour.
    """
    gc.collect()
    start = perf_counter()
    values = _tree_greeks(params, vol_shift).all()
    black, reason = _black_greeks(params)
    nodes = _node_delta_gamma(params)
    corrected = _control_variate(params, values, vol_shift)
    return {"values": values, "table": _greek_table(values, black, nodes, corrected),
            "black_reason": reason, "seconds": perf_counter() - start}


# Delta et gamma pour une liste de spots (profil utile à la couverture)
def greeks_by_spot(params: Parameters, spots: list[float]) -> pd.DataFrame:
    """Prix, delta et gamma de l'arbre pour chaque spot de la liste."""
    values = [_finite_number("Le spot", spot)
              for spot in _study_values("La liste des spots", spots, MAX_STRIKES)]
    if any(spot <= 0 for spot in values):
        raise ValueError("Les spots doivent être strictement positifs.")
    rows = []
    # Un objet TreeGreeks par spot : 3 arbres (S/α, S, S×α) chacun.
    for spot in values:
        greeks = _tree_greeks(replace(params, spot=spot), 0.01)
        rows.append({"spot": spot, "price": greeks.price(spot, params.vol, 0),
                     "delta": greeks.delta(), "gamma": greeks.gamma()})
    return pd.DataFrame(rows)


# Variation de valeur : approximation par les grecques et vrai repricing
def taylor_check(params: Parameters, greeks: dict[str, float], spot_move: float,
                 vol_move_points: float, days: int) -> dict[str, float]:
    """Comparer le P&L de Taylor au prix recalculé après le choc.

    Le choc : spot + ``spot_move`` €, vol + ``vol_move_points`` points,
    date de calcul avancée de ``days`` jours (maturité inchangée).
    """
    shocked = replace(params, spot=params.spot + spot_move,
                      vol=params.vol + vol_move_points / 100.0,
                      pricing_date=params.pricing_date + dt.timedelta(days=days))
    result = taylor_pnl(greeks, spot_move, vol_move_points, days)
    tree = None
    try:
        tree = make_tree(shocked)
        result["repriced"] = _price(tree) - greeks["price"]
    finally:
        del tree
        gc.collect()
    # P&L d'un portefeuille option − Δ actions, recalculé exactement.
    result["repriced_hedged"] = result["repriced"] - greeks["delta"] * spot_move
    return result
