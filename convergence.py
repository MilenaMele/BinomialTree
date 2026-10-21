# =============================================================================
# Convergence de l'arbre vers Black-Scholes.
#
# Fournit le prix Black-Scholes européen (avec rendement continu q).
# Pas de dividende cash ici, sinon Black-Scholes n'est plus une référence exacte.
# =============================================================================
"""Convergence de l'arbre vers Black-Scholes."""

# Imports : bibliothèque standard, paquets externes puis modules du projet.
import datetime as dt
import gc
from math import exp, log, pi, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from arbre_binomial import Market, Model, Option, Tree


# Référence Black-Scholes et étude de l'écart arbre − Black selon N
class Convergence:
    """Référence Black-Scholes et étude de l'écart arbre − Black selon N."""

    def __init__(self, market: Market, option: Option, pricing_date: dt.date) -> None:
        # Black-Scholes n'est exact que pour une européenne sans cash.
        if option.exercise_type.lower() != "european":
            raise ValueError("Black-Scholes ne sert de référence que pour une option européenne.")
        if market.dividend_amount != 0.0:
            raise ValueError("Avec un dividende cash, Black-Scholes n'est pas une référence "
                             "exacte.")
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.maturity = (option.maturity - pricing_date).days / 365.0
        if self.maturity <= 0:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        # Les fichiers de l'étude sont écrits à côté de ce script.
        self.output = Path(__file__).resolve().parent
        self.results: np.ndarray | None = None

    # Vrai pour un call, faux pour un put
    def _is_call(self) -> bool:
        """Vrai pour un call, faux pour un put."""
        return self.option.type.lower() == "call"

    # Cas limites K=0 et sigma=0, traités avant les divisions par K ou sigma
    def _limit_price(self, strike: float) -> float | None:
        """Cas limites K=0 et sigma=0, traités avant les divisions par K ou sigma."""
        spot, rate = self.market.spot, self.market.rate
        q, maturity = self.market.dividend_yield, self.maturity
        if strike == 0:
            return spot * exp(-q * maturity) if self._is_call() else 0.0
        if self.market.vol != 0:
            return None
        # Volatilité nulle : le sous-jacent suit son forward sans aléa.
        terminal_spot = spot * exp((rate - q) * maturity)
        if self._is_call():
            payoff = max(terminal_spot - strike, 0.0)
        else:
            payoff = max(strike - terminal_spot, 0.0)
        return exp(-rate * maturity) * payoff

    # Prix européen avec rendement continu ; pas d'approximation cash
    def black_price(self, strike: float) -> float:
        """Prix européen avec rendement continu ; pas d'approximation cash."""
        if not np.isfinite(strike) or strike < 0:
            raise ValueError("Le strike doit être fini et positif ou nul.")
        limit = self._limit_price(strike)
        if limit is not None:
            return limit
        # Formule de Black-Scholes avec rendement de dividende continu q.
        S, r, q = self.market.spot, self.market.rate, self.market.dividend_yield
        sigma, T = self.market.vol, self.maturity
        d1 = (log(S / strike) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        N = NormalDist().cdf
        if self._is_call():
            return S * exp(-q * T) * N(d1) - strike * exp(-r * T) * N(d2)
        return strike * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)

    # Constante C du « triangle rouge » : écart ≈ C / N
    def triangle_constant(self) -> float:
        """Constante C du « triangle rouge » : écart ≈ C / N.

        L'arbre ne voit pas le coude du payoff en K et perd l'aire d'un petit
        triangle entre deux nœuds. Cette aire est proportionnelle à Δt = T/N.
        """
        r, q = self.market.rate, self.market.dividend_yield
        sigma, T = self.market.vol, self.maturity
        if sigma == 0:
            return 0.0
        forward = self.market.spot * exp((r - q) * T)
        return (exp(-r * T) * forward * sigma**2 * T
                / (2 * sqrt(2 * pi) * sqrt(exp(sigma**2 * T) - 1)))

    # Construire et pricer un arbre à N pas ; renvoie (prix, secondes)
    def _timed_price(self, n: int) -> tuple[float, float]:
        """Construire et pricer un arbre à N pas ; renvoie (prix, secondes)."""
        gc.collect()  # on vide la mémoire avant de chronométrer
        start = perf_counter()
        tree = Tree(Model(self.pricing_date, n), self.option, self.market)
        tree.build()
        price = tree.price()
        return price, perf_counter() - start

    # Pour chaque N : prix de l'arbre, écart avec Black-Scholes et temps
    def run(self, steps: list[int]) -> None:
        """Pour chaque N : prix de l'arbre, écart avec Black-Scholes et temps."""
        if not steps:
            raise ValueError("Fournir au moins un nombre de pas.")
        # Valider toute la série avant le premier calcul ou fichier créé.
        for n in steps:
            Model(self.pricing_date, n)
        self.output.mkdir(parents=True, exist_ok=True)
        black = self.black_price(self.option.strike)
        rows = []
        print("Calcul de la convergence en cours...")
        for n in steps:
            price, elapsed = self._timed_price(n)
            rows.append((n, price, price - black, elapsed))
        self.results = np.array(rows)
        np.savetxt(self.output / "convergence_resultats.txt", self.results,
                   fmt=["%d", "%.10f", "%.10f", "%.6f"],
                   header="N  prix_arbre  ecart  temps_secondes")
        print(f"Prix Black-Scholes : {black:.6f}")
        print(f"Prix arbre avec {steps[-1]} pas : {rows[-1][1]:.6f}")
        print(f"Écart : {rows[-1][2]:.6f}")
        print(f"Constante du triangle C = {self.triangle_constant():.4f}")

    # Trois graphiques : prix, écart × N et temps de calcul
    def plot(self) -> None:
        """Trois graphiques : prix, écart × N et temps de calcul."""
        if self.results is None:
            raise RuntimeError("Appeler run(steps) avant plot().")
        n, prices, errors, times = self.results.T
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        # 1. Le prix de l'arbre se rapproche du prix Black-Scholes
        axes[0].plot(n, prices, ".", markersize=3, label="Arbre")
        axes[0].axhline(self.black_price(self.option.strike), color="red", label="Black-Scholes")
        axes[0].set(title="Prix en fonction de N", xlabel="N", ylabel="Prix")
        # 2. Écart × N : si ça reste borné, l'écart diminue en 1/N
        axes[1].plot(n, errors * n, ".", markersize=3)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set(title="(Prix arbre − Black) × N", xlabel="N", ylabel="Écart × N")
        # 3. Temps de calcul en log-log, comparé à N²
        axes[2].loglog(n, times, ".", markersize=3, label="Temps mesuré")
        axes[2].loglog(n, times[-1] * (n / n[-1])**2, "--", label="N²")
        axes[2].set(title="Temps de calcul", xlabel="N", ylabel="Secondes")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[2].legend()
        fig.savefig(self.output / "convergence.png", dpi=150)

    # Écart arbre − Black en fonction du strike, pour N = 10 et 100
    def plot_strike(self) -> None:
        """Écart arbre − Black en fonction du strike, pour N = 10 et 100."""
        strikes = np.linspace(80, 120, 161)
        black_prices = np.array([self.black_price(k) for k in strikes])
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for n in (10, 100):
            # L'arbre ne dépend pas du strike : on le construit une seule fois
            option = Option(100, self.option.maturity, self.option.type, "European")
            tree = Tree(Model(self.pricing_date, n), option, self.market).build()
            prices = []
            for k in strikes:
                option.strike = k
                prices.append(tree.price())
            ax.plot(strikes, np.array(prices) - black_prices, label=f"N = {n}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set(title="Écart arbre − Black en fonction du strike", xlabel="Strike K",
               ylabel="Écart")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(self.output / "convergence_strike.png", dpi=150)


# Étude complète : tous les N jusqu'à 300 puis par paquets jusqu'à 1000
def main() -> None:
    """Étude complète : tous les N jusqu'à 300 puis par paquets jusqu'à 1000."""
    option = Option(strike=101, maturity=dt.date(2027, 9, 2), type="Call",
                    exercise_type="European")
    market = Market(spot=100, rate=0.03, vol=0.20, dividend_yield=0.02)
    study = Convergence(market, option, dt.date(2026, 9, 15))
    # Tous les N jusqu'à 300, puis un N impair et un N pair tous les 50 pas
    steps = list(range(1, 301))
    for n in range(350, 1001, 50):
        steps += [n - 1, n]
    study.run(steps)
    study.plot()
    study.plot_strike()
    plt.show()


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Lecture des courbes (cours p24 et suivantes)
# -----------------------------------------------------------------------------
# Le payoff possède un coude au strike. Sa position entre deux nœuds change
# lorsque N varie : l'erreur peut osciller et n'est pas monotone. Une bande
# bornée pour N*(prix_arbre-prix_Black) est compatible avec un ordre 1/N ;
# cela ne signifie pas que chaque erreur individuelle est exactement C/N.
#
# Sans pruning, il y a (N+1)*(N+2)/2 nœuds dans le cas binomial standard.
# Le temps et la mémoire sont donc d'ordre N². L'étude de convergence garde
# un seuil nul pour isoler l'erreur de discrétisation.
