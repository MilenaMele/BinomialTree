# =============================================================================
# Grecques de l'option : delta, gamma, vega, vomma, vanna et theta.
#
# À quoi sert ce module ?
#     Il calcule les sensibilités du prix donné par l'arbre, comme dans le
#     cours (p. 17) : on passe une FONCTION DE PRICING en paramètre à une
#     classe de dérivation numérique (``OneDimDerivative``), qui fait des
#     différences finies centrées en décalant un paramètre de ``shift``.
#
#     * delta, gamma : dérivées 1re et 2e par rapport au spot S0 ;
#     * vega, vomma  : dérivées 1re et 2e par rapport à la volatilité σ ;
#     * vanna        : dérivée croisée ∂²V/∂S∂σ = dérivée du delta en σ ;
#     * theta        : variation du prix quand un jour passe (maturité fixe).
#
#     Il donne aussi l'« autre astuce » du cours pour delta et gamma : lire
#     les valeurs de l'option sur trois nœuds voisins, sans reconstruire
#     d'arbre, et les grecques exactes de Black-Scholes pour comparaison.
#
# Chaque prix décalé reconstruit un arbre complet : c'est du Python pur,
# sans Streamlit, utilisable seul (``python grecques.py``).
# =============================================================================
"""Grecques de l'option par différences finies sur l'arbre (cours p. 17)."""

# Imports : bibliothèque standard, numpy puis modules du projet.
from __future__ import annotations

import datetime as dt
import gc
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Callable

import numpy as np

from arbre_binomial import Market, Model, Option, Tree

# Décalage par défaut de la volatilité : 1 point (σ ± 1 %). Le spot est décalé
# d'un cran de la grille (S × α et S / α), comme les nœuds voisins de la racine.
DEFAULT_VOL_SHIFT = 0.01
# Unités affichées : vega, vomma et vanna sont exprimés pour 1 point de vol.
VOL_POINT = 0.01
GREEK_NAMES = ("delta", "gamma", "vega", "vomma", "vanna", "theta")


# -----------------------------------------------------------------------------
# DÉRIVATION NUMÉRIQUE : une fonction est passée en paramètre (cours p. 17).
# -----------------------------------------------------------------------------
# Dérivées 1re et 2e d'une fonction f(paramètres, x) par différences centrées
class OneDimDerivative:
    """Dérivées 1re et 2e de ``function(other_parameters, x)`` en x.

    Reprend la classe du cours : la fonction de pricing est un paramètre,
    ``other_parameters`` contient tout ce qui ne bouge pas, ``shift`` est h.
    """

    def __init__(self, function: Callable[[object, float], float],
                 other_parameters: object, shift: float = 1) -> None:
        # Un décalage nul ou négatif rendrait la division impossible.
        if not shift > 0:
            raise ValueError("Le décalage (shift) doit être strictement positif.")
        self.f = function
        self.param = other_parameters
        self.shift = shift

    def _value(self, x: float) -> float:
        """Évaluer la fonction en x, les autres paramètres étant fixés."""
        return self.f(self.param, x)

    # Dérivée première : sert au delta, au vega et au vanna.
    def first(self, x: float) -> float:
        """Dérivée 1re centrée : (f(x+h) − f(x−h)) / 2h."""
        return (self._value(x + self.shift) - self._value(x - self.shift)) / (2 * self.shift)

    # Dérivée seconde : sert au gamma et au vomma.
    def second(self, x: float) -> float:
        """Dérivée 2e centrée : (f(x+h) − 2 f(x) + f(x−h)) / h²."""
        up, mid, down = (self._value(x + self.shift), self._value(x),
                         self._value(x - self.shift))
        return (up - 2 * mid + down) / self.shift**2


# -----------------------------------------------------------------------------
# FONCTIONS DE PRICING passées aux classes de dérivation.
# -----------------------------------------------------------------------------
# Prix de l'arbre pour un log-spot x = ln(S) (les autres paramètres sont fixés)
def _price_for_log_spot(greeks: TreeGreeks, log_spot: float) -> float:
    """Fonction f(paramètres, ln S) utilisée pour delta et gamma."""
    return greeks.price(exp(log_spot), greeks.market.vol, 0)


# Prix de l'arbre pour une volatilité donnée
def _price_for_vol(greeks: TreeGreeks, vol: float) -> float:
    """Fonction f(paramètres, σ) utilisée pour vega et vomma."""
    return greeks.price(greeks.market.spot, vol, 0)


# Delta de l'arbre pour une volatilité donnée (sert au vanna)
def _delta_for_vol(greeks: TreeGreeks, vol: float) -> float:
    """Fonction f(paramètres, σ) = delta : le vanna est sa dérivée en σ."""
    return greeks.delta_at_vol(vol)


# -----------------------------------------------------------------------------
# GRECQUES DE L'ARBRE
# -----------------------------------------------------------------------------
# Calcul de toutes les grecques d'une option par différences finies sur l'arbre
class TreeGreeks:
    """Grecques d'une option : chaque prix décalé reconstruit un arbre.

    Le spot est décalé en logarithme d'un cran de grille, ln(α) : les arbres
    S×α et S/α ont alors les mêmes nœuds que l'arbre initial, décalés d'un
    rang (ce sont les « nœuds autour de la racine » du cours). Sans cela, la
    position du strike entre deux nœuds change et le gamma devient faux.
    Les prix déjà calculés sont mémorisés (dictionnaire de NOMBRES, clé =
    spot, vol, décalage en jours) : gamma réutilise ainsi les prix du delta.
    """

    def __init__(self, market: Market, option: Option, model: Model,
                 vol_shift: float = DEFAULT_VOL_SHIFT) -> None:
        # vol_shift est absolu (0.01 = 1 point de volatilité).
        self.market = market
        self.option = option
        self.model = model
        self.vol_shift = vol_shift
        self.log_spot_shift = self._grid_log_step()
        self._cache: dict[tuple[float, float, int], float] = {}
        self._check_shifts()

    def _grid_log_step(self, vol: float | None = None) -> float:
        """ln(α) : écart en log entre deux voisins d'une colonne de l'arbre."""
        vol = self.market.vol if vol is None else vol
        if vol == 0:
            return 0.01
        market = Market(self.market.spot, self.market.rate, vol, self.market.dividend_yield,
                        self.market.dividend_amount, self.market.ex_dividend_date)
        # Tree.__init__ calcule α sans construire les nœuds : c'est immédiat.
        return float(log(Tree(self.model, self.option, market).alpha))

    def _check_shifts(self) -> None:
        """La volatilité décalée vers le bas doit rester positive."""
        if not 0 < self.vol_shift:
            raise ValueError("Le décalage de volatilité doit être strictement positif.")
        if self.market.vol - self.vol_shift < 0:
            raise ValueError("La volatilité doit dépasser son décalage pour vega et vomma.")

    def _build_price(self, spot: float, vol: float, days: int) -> float:
        """Construire et pricer un nouvel arbre avec le marché décalé."""
        market = Market(spot, self.market.rate, vol, self.market.dividend_yield,
                        self.market.dividend_amount, self.market.ex_dividend_date)
        pricing_date = self.model.pricing_date + dt.timedelta(days=days)
        model = Model(pricing_date, self.model.nb_steps, self.model.pruning_threshold)
        option = Option(self.option.strike, self.option.maturity, self.option.type,
                        self.option.exercise_type)
        tree = Tree(model, option, market).build()
        value = tree.price()
        # Les liens entre nœuds forment des cycles : libérer l'arbre tout de suite.
        del tree
        gc.collect()
        return float(value)

    def price(self, spot: float, vol: float, days: int) -> float:
        """Prix pour (spot, vol), la date de calcul étant avancée de ``days``."""
        key = (float(spot), float(vol), int(days))
        if key not in self._cache:
            self._cache[key] = self._build_price(*key)
        return self._cache[key]

    # Dérivées par rapport au spot : delta et gamma (via x = ln S)
    def _log_spot_derivative(self) -> OneDimDerivative:
        """Dérivation en x = ln S avec un pas d'un cran de grille."""
        return OneDimDerivative(_price_for_log_spot, self, self.log_spot_shift)

    def delta(self) -> float:
        """∂V/∂S = (1/S) ∂V/∂x : nombre d'actions pour couvrir une option."""
        spot = self.market.spot
        return self._log_spot_derivative().first(log(spot)) / spot

    # Delta à une autre volatilité : chaque σ a sa propre grille (α dépend de σ).
    def delta_at_vol(self, vol: float) -> float:
        """Delta calculé avec une autre volatilité (et la grille de cet arbre)."""
        spot, step = self.market.spot, self._grid_log_step(vol)
        up = self.price(spot * exp(step), vol, 0)
        down = self.price(spot * exp(-step), vol, 0)
        return (up - down) / (2 * step) / spot

    def gamma(self) -> float:
        """∂²V/∂S² = (∂²V/∂x² − ∂V/∂x) / S² : variation du delta pour 1 €."""
        spot = self.market.spot
        derivative = self._log_spot_derivative()
        x = log(spot)
        return (derivative.second(x) - derivative.first(x)) / spot**2

    # Dérivées par rapport à la volatilité (exprimées pour 1 point de vol)
    def vega(self) -> float:
        """∂V/∂σ, pour une hausse de 1 point de volatilité (σ + 1 %)."""
        derivative = OneDimDerivative(_price_for_vol, self, self.vol_shift)
        return derivative.first(self.market.vol) * VOL_POINT

    def vomma(self) -> float:
        """∂²V/∂σ², pour des variations de 1 point de volatilité."""
        derivative = OneDimDerivative(_price_for_vol, self, self.vol_shift)
        return derivative.second(self.market.vol) * VOL_POINT**2

    def vanna(self) -> float:
        """∂²V/∂S∂σ : variation du delta pour +1 point de volatilité."""
        # Chaque delta utilise la grille de SON arbre (σ ± h), d'où un vanna lisse.
        derivative = OneDimDerivative(_delta_for_vol, self, self.vol_shift)
        return derivative.first(self.market.vol) * VOL_POINT

    # Dérivée par rapport au temps : un jour passe, la maturité reste fixe
    def theta(self) -> float:
        """V(t + 1 jour) − V(t) : gain (ou perte) de valeur en un jour, en €."""
        if (self.option.maturity - self.model.pricing_date).days < 2:
            raise ValueError("Le theta sur un jour demande au moins deux jours de vie.")
        today = self.price(self.market.spot, self.market.vol, 0)
        return self.price(self.market.spot, self.market.vol, 1) - today

    def all(self) -> dict[str, float]:
        """Prix et six grecques dans un dictionnaire de nombres."""
        values = {"price": self.price(self.market.spot, self.market.vol, 0)}
        for name in GREEK_NAMES:
            values[name] = getattr(self, name)()
        return values


# -----------------------------------------------------------------------------
# AUTRE ASTUCE DU COURS : delta et gamma lus sur trois nœuds de l'arbre.
# -----------------------------------------------------------------------------
# Trois nœuds voisins (haut, milieu, bas) de la colonne du pas 2
def _three_nodes(tree: Tree) -> tuple:
    """Tronc du pas 2 et ses deux voisins, atteints par les liens directs."""
    trunk = tree.root.next_trunk.next_trunk if tree.model.nb_steps >= 2 else None
    if trunk is None or trunk.above is None or trunk.below is None:
        raise ValueError("Il faut au moins 2 pas et 3 nœuds au pas 2 pour cette méthode.")
    return trunk.above, trunk, trunk.below


# Delta et gamma déduits des valeurs déjà calculées dans l'arbre
def tree_node_delta_gamma(tree: Tree) -> tuple[float, float]:
    """Delta et gamma sans reconstruire d'arbre (cours p. 17, « autre astuce »).

    Le cours ajoute des nœuds autour de la racine ; ici on lit les trois
    nœuds du pas 2, qui entourent le spot. C'est une estimation à t = 2Δt,
    très proche de t = 0 quand N est grand. Aucun vega/vomma par ce moyen.
    """
    tree.price()
    up, mid, down = _three_nodes(tree)
    s_up, s_mid, s_down = up.spot_price, mid.spot_price, down.spot_price
    v_up, v_mid, v_down = up.option_value, mid.option_value, down.option_value
    delta = (v_up - v_down) / (s_up - s_down)
    # Gamma : variation des deux pentes, divisée par la demi-distance.
    slope_up = (v_up - v_mid) / (s_up - s_mid)
    slope_down = (v_mid - v_down) / (s_mid - s_down)
    gamma = (slope_up - slope_down) / ((s_up - s_down) / 2)
    return float(delta), float(gamma)


# -----------------------------------------------------------------------------
# RÉFÉRENCE : grecques exactes de Black-Scholes (européenne, sans cash).
# -----------------------------------------------------------------------------
# Quantités d1, d2 et facteurs d'actualisation de Black-Scholes
def _black_terms(market: Market, strike: float, maturity: float) -> dict[str, float]:
    """d1, d2, densité n(d1) et actualisations exp(−qT), exp(−rT)."""
    sigma, sqrt_t = market.vol, sqrt(maturity)
    d1 = ((log(market.spot / strike) + (market.rate - market.dividend_yield
                                         + sigma**2 / 2) * maturity) / (sigma * sqrt_t))
    d2 = d1 - sigma * sqrt_t
    return {"d1": d1, "d2": d2, "n_d1": NormalDist().pdf(d1), "sqrt_t": sqrt_t,
            "eq": exp(-market.dividend_yield * maturity), "er": exp(-market.rate * maturity)}


# Theta annuel de Black-Scholes (∂V/∂t, le temps qui passe)
def _black_theta_year(market: Market, strike: float, terms: dict[str, float],
                      is_call: bool) -> float:
    """Theta par an : terme de volatilité + termes de taux et de dividende."""
    N = NormalDist().cdf
    sign = 1 if is_call else -1
    decay = -market.spot * terms["eq"] * terms["n_d1"] * market.vol / (2 * terms["sqrt_t"])
    carry = (-sign * market.rate * strike * terms["er"] * N(sign * terms["d2"])
             + sign * market.dividend_yield * market.spot * terms["eq"]
             * N(sign * terms["d1"]))
    return decay + carry


# Les six grecques de Black-Scholes, dans les mêmes unités que l'arbre
def black_scholes_greeks(market: Market, option: Option,
                         pricing_date: dt.date) -> dict[str, float]:
    """Grecques exactes d'une européenne avec rendement continu q, sans cash."""
    maturity = (option.maturity - pricing_date).days / 365.0
    if market.vol <= 0 or option.strike <= 0 or maturity <= 0:
        raise ValueError("Black-Scholes demande σ > 0, K > 0 et une maturité future.")
    terms = _black_terms(market, option.strike, maturity)
    is_call = option.type.lower() == "call"
    N = NormalDist().cdf
    delta = terms["eq"] * (N(terms["d1"]) if is_call else N(terms["d1"]) - 1)
    vega_unit = market.spot * terms["eq"] * terms["n_d1"] * terms["sqrt_t"]
    # Mêmes conventions que TreeGreeks : vol en points, theta par jour.
    return {
        "delta": delta,
        "gamma": terms["eq"] * terms["n_d1"] / (market.spot * market.vol * terms["sqrt_t"]),
        "vega": vega_unit * VOL_POINT,
        "vomma": vega_unit * terms["d1"] * terms["d2"] / market.vol * VOL_POINT**2,
        "vanna": -terms["eq"] * terms["n_d1"] * terms["d2"] / market.vol * VOL_POINT,
        "theta": _black_theta_year(market, option.strike, terms, is_call) / 365.0,
    }


# -----------------------------------------------------------------------------
# COUVERTURE : approximation de Taylor de la variation de valeur.
# -----------------------------------------------------------------------------
# Variation de prix estimée à partir des grecques (développement de Taylor)
def taylor_pnl(greeks: dict[str, float], spot_move: float, vol_move_points: float,
               days: int) -> dict[str, float]:
    """dV ≈ Δ dS + ½ Γ dS² + vega dσ + ½ vomma dσ² + vanna dS dσ + θ jours.

    ``vol_move_points`` est en points de vol (2 = +2 %). Renvoie chaque terme
    et le P&L d'un portefeuille delta-neutre (option − Δ actions).
    """
    terms = {
        "delta": greeks["delta"] * spot_move,
        "gamma": 0.5 * greeks["gamma"] * spot_move**2,
        "vega": greeks["vega"] * vol_move_points,
        "vomma": 0.5 * greeks["vomma"] * vol_move_points**2,
        "vanna": greeks["vanna"] * spot_move * vol_move_points,
        "theta": greeks["theta"] * days,
    }
    terms["total"] = float(np.sum(list(terms.values())))
    # La couverture delta annule le terme du premier ordre en spot.
    terms["hedged_total"] = terms["total"] - terms["delta"]
    return terms


# -----------------------------------------------------------------------------
# Lancement autonome (sans Streamlit) : comparaison arbre / Black-Scholes.
# -----------------------------------------------------------------------------
def main() -> None:
    """Grecques d'un call européen : arbre (N=400) et Black-Scholes."""
    market = Market(spot=100, rate=0.03, vol=0.20, dividend_yield=0.02)
    option = Option(101, dt.date(2027, 9, 2), "Call", "European")
    model = Model(dt.date(2026, 9, 15), 400)
    tree_values = TreeGreeks(market, option, model).all()
    black = black_scholes_greeks(market, option, model.pricing_date)
    # Une ligne par grecque : arbre, Black-Scholes et écart.
    print(f"{'Grecque':<8}{'Arbre':>14}{'Black-Scholes':>16}{'Écart':>12}")
    for name in GREEK_NAMES:
        gap = tree_values[name] - black[name]
        print(f"{name:<8}{tree_values[name]:>14.6f}{black[name]:>16.6f}{gap:>12.2e}")


if __name__ == "__main__":
    main()
