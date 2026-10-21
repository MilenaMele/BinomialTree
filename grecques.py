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
# Chaque prix décalé reconstruit un arbre complet : c'est du Python pur,
# sans Streamlit, utilisable seul (``python grecques.py``).
# =============================================================================
"""Grecques de l'option par différences finies sur l'arbre (cours p. 17)."""

# Imports : bibliothèque standard puis moteur de l'arbre.
from __future__ import annotations

import datetime as dt
import gc
from math import exp, log
from typing import Callable

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
# Lancement autonome (sans Streamlit) : grecques d'un exemple.
# -----------------------------------------------------------------------------
def main() -> None:
    """Prix et grecques d'un call européen (N = 400), affichés dans le terminal."""
    market = Market(spot=100, rate=0.03, vol=0.20, dividend_yield=0.02)
    option = Option(101, dt.date(2027, 9, 2), "Call", "European")
    values = TreeGreeks(market, option, Model(dt.date(2026, 9, 15), 400)).all()
    # Une ligne par grandeur.
    for name, value in values.items():
        print(f"{name:<8}{value:>14.6f}")


if __name__ == "__main__":
    main()
