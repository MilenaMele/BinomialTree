# =============================================================================
# Moteur de pricing : arbre recombiné à liens directs (cœur du projet).
#
# Construit l'arbre du sous-jacent (binomial, avec un pas trinomial
# au détachement d'un dividende cash), puis valorise une option
# call/put européenne ou américaine en remontant l'arbre. Il calcule
# aussi les profils par date du cours: valeur moyenne
# actualisée et probabilités d'exercice.
#
# =============================================================================
"""Moteur de pricing : arbre recombiné à liens directs (cœur du projet)."""

# Imports : bibliothèque standard, paquets externes puis modules du projet.
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterator

import numpy as np

# Noms des trois branches possibles d'un nœud (chaînes, jamais des nœuds).
BRANCHES = ("up", "mid", "down")


# -----------------------------------------------------------------------------
# RÉSULTATS PAR DATE : données de sortie, jamais un moyen d'accès aux nœuds.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExerciseProfileRow:
    """Une ligne du profil risque-neutre calculé par ``exercise_profile``.

    Les valeurs monétaires sont toutes actualisées à la date de calcul.
    ``date`` peut contenir une heure : les N intervalles ont la même durée
    et ne correspondent pas nécessairement à un nombre entier de jours.
    ``survival_probability`` est mesurée APRÈS la décision de la date ; au
    dernier pas, elle représente les options expirant sans paiement.
    """

    # Repère temporel de la ligne.
    step: int
    time_years: float
    date: dt.datetime
    # Valeur moyenne et probabilités d'exercice de la date.
    discounted_mean_value: float
    exercise_region_probability: float
    first_exercise_probability: float
    cumulative_exercise_probability: float
    survival_probability: float
    # Contrôle de conservation de la valeur (vivants + flux payés).
    discounted_alive_value: float
    discounted_exercise_cashflow: float
    cumulative_discounted_cashflow: float
    total_value: float


# -----------------------------------------------------------------------------
# MODEL : choix numériques (date de calcul, nombre de pas, élagage).
# -----------------------------------------------------------------------------
class Model:
    """Paramètres numériques : date de calcul, nombre de pas et élagage."""

    def __init__(self, pricing_date: dt.date, nb_steps: int,
                 pruning_threshold: float = 0.0) -> None:
        # On valide tout avant d'enregistrer, pour échouer tôt et clairement.
        self._validate(pricing_date, nb_steps, pruning_threshold)
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        # Pruning : un nœud du bord dont la probabilité d'être atteint est sous
        # ce seuil n'a plus qu'un seul enfant (0 = pas de pruning).
        self.pruning_threshold = pruning_threshold

    # Refuser une date avec heure, un N non entier ou un seuil hors [0, 1]
    @staticmethod
    def _validate(pricing_date: dt.date, nb_steps: int, threshold: float) -> None:
        """Refuser une date avec heure, un N non entier ou un seuil hors [0, 1]."""
        # datetime hérite de date : on l'exclut explicitement.
        if not isinstance(pricing_date, dt.date) or isinstance(pricing_date, dt.datetime):
            raise ValueError("pricing_date doit être une date (datetime.date).")
        # bool hérite de int : True ne doit pas valoir « 1 pas ».
        if isinstance(nb_steps, bool) or not isinstance(nb_steps, int) or nb_steps < 1:
            raise ValueError("nb_steps doit être un entier strictement positif.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Le seuil de pruning doit être fini et compris entre 0 et 1.")


# -----------------------------------------------------------------------------
# OPTION : le contrat (payoff call/put, exercice européen/américain).
# -----------------------------------------------------------------------------
class Option:
    """Contrat : payoff call/put et possibilité d'exercice anticipé."""

    def __init__(self, strike: float, maturity: dt.date, type: str,
                 exercise_type: str) -> None:
        # Le nom ``type`` est historique : on le garde pour ne casser aucun appel.
        self._validate(strike, maturity, type, exercise_type)
        self.strike = strike
        self.maturity = maturity
        self.type = type
        self.exercise_type = exercise_type

    # Vérifier strike, date et libellés avant toute construction
    @staticmethod
    def _validate(strike: float, maturity: dt.date, kind: str, exercise: str) -> None:
        """Vérifier strike, date et libellés avant toute construction."""
        if not np.isfinite(strike) or strike < 0:
            raise ValueError("Le strike doit être fini et positif ou nul.")
        if not isinstance(maturity, dt.date) or isinstance(maturity, dt.datetime):
            raise ValueError("maturity doit être une date (datetime.date).")
        # Les libellés sont comparés sans tenir compte des majuscules.
        if kind.lower() not in ("call", "put"):
            raise ValueError("Option type must be 'Call' or 'Put'")
        if exercise.lower() not in ("european", "american"):
            raise ValueError("Exercise type must be 'European' or 'American'")

    # Valeur d'exercice immédiat (payoff) pour un prix du sous-jacent
    def intrinsic_value(self, spot_price: float) -> float:
        """Valeur d'exercice immédiat (payoff) pour un prix du sous-jacent."""
        if self.type.lower() == "call":
            return max(spot_price - self.strike, 0.0)
        if self.type.lower() == "put":
            return max(self.strike - spot_price, 0.0)
        # Cas impossible après validation, gardé par sécurité.
        raise ValueError("Option type must be 'Call' or 'Put'")


# -----------------------------------------------------------------------------
# MARKET : spot, taux, volatilité et dividendes (continu q et cash D).
# -----------------------------------------------------------------------------
class Market:
    """Marché constant : spot, taux r, volatilité, rendement q et un cash D."""

    def __init__(self, spot: float, rate: float, vol: float,
                 dividend_yield: float = 0.0, dividend_amount: float = 0.0,
                 ex_dividend_date: dt.date | None = None) -> None:
        # Validation groupée dans une méthode privée.
        self._validate(spot, rate, vol, dividend_yield, dividend_amount, ex_dividend_date)
        self.spot = spot
        self.rate = rate
        self.vol = vol
        # q est un rendement annuel (0.02 = 2 %), D un montant en euros.
        self.dividend_yield = dividend_yield
        self.dividend_amount = dividend_amount
        self.ex_dividend_date = ex_dividend_date

    # Refuser les valeurs non finies, négatives ou un cash sans date
    @staticmethod
    def _validate(spot: float, rate: float, vol: float, q: float, cash: float,
                  ex_date: dt.date | None) -> None:
        """Refuser les valeurs non finies, négatives ou un cash sans date."""
        if not all(np.isfinite(x) for x in (spot, rate, vol, q, cash)):
            raise ValueError("Les paramètres de marché doivent être finis.")
        # Le taux r peut être négatif ; les autres grandeurs non.
        if spot <= 0 or vol < 0 or q < 0 or cash < 0:
            raise ValueError("Il faut spot > 0, vol >= 0, q >= 0 et dividende cash >= 0.")
        if cash > 0 and ex_date is None:
            raise ValueError("Un dividende cash positif nécessite sa date de détachement.")
        if ex_date is not None and (
            not isinstance(ex_date, dt.date) or isinstance(ex_date, dt.datetime)
        ):
            raise ValueError("ex_dividend_date doit être une date (datetime.date).")


# -----------------------------------------------------------------------------
# NODE : un état du sous-jacent à une date, relié à ses voisins et enfants.
# -----------------------------------------------------------------------------
class Node:
    """Un nœud conserve son prix, ses liens et les résultats du pricing."""

    def __init__(self, tree: Tree, spot_price: float, step: int) -> None:
        self.tree = tree
        self.spot_price = spot_price
        self.step = step
        # Enfants à la date suivante et probabilités de transition.
        self.next_up: Node | None = None
        self.next_down: Node | None = None
        self.next_mid: Node | None = None
        self.p_up = tree.p_up
        self.p_down = tree.p_down
        self.p_mid = 0.0
        # Voisins dans la même colonne (même date).
        self.above: Node | None = None
        self.below: Node | None = None
        self.option_value: float | None = None
        # Continuation = valeur actualisée des enfants, AVANT comparaison
        # avec le payoff. À maturité, il n'y a plus de continuation (None).
        self.continuation_value: float | None = None
        self.exercise_now = False
        # Ce drapeau décrit la construction, indépendamment de l'option.
        # Une transition déterministe (volatilité nulle) n'est pas un pruning.
        self.is_pruned = False
        # Masse qui atteint le nœud sans exercice antérieur ; remise à zéro
        # à chaque calcul du profil. Elle ne remplace jamais ``proba``.
        self.alive_proba = 0.0
        # Probabilité d'atteindre ce nœud depuis la racine (sert au pruning).
        self.proba = 0.0

    # Espérance conditionnelle sur le prochain pas, dividendes inclus
    def forward(self) -> float:
        """Espérance conditionnelle sur le prochain pas, dividendes inclus."""
        return self.tree.expected_spot(self.spot_price, self.step)

    # Variance conditionnelle du prochain pas (approximation du cours)
    def variance(self) -> float:
        """Variance conditionnelle du prochain pas (approximation du cours)."""
        return self.tree.conditional_variance(self.spot_price, self.step)

    # Somme p_branche × V_enfant sur les branches existantes
    def _expected_child_value(self) -> float:
        """Somme p_branche × V_enfant sur les branches existantes."""
        expectation = 0.0
        for branch in BRANCHES:
            child = getattr(self, "next_" + branch)
            # Une branche absente (pruning, pas binomial) ne contribue pas.
            if child is not None:
                expectation += getattr(self, "p_" + branch) * child.option_value
        return expectation

    # Rétropropager puis enregistrer la décision de gestion optimale
    def compute_option_value(self) -> None:
        """Rétropropager puis enregistrer la décision de gestion optimale.

        Les valeurs des enfants sont déjà calculées puisque ``Tree.price``
        parcourt les dates à rebours. Le facteur d'actualisation utilise r,
        même lorsque le rendement du sous-jacent est r-q.
        """
        self.continuation_value = float(self.tree.df * self._expected_child_value())
        self.exercise_now = False
        exercise_type = self.tree.option.exercise_type.lower()
        # Européenne : on ne peut que conserver le contrat.
        if exercise_type == "european":
            self.option_value = self.continuation_value
            return
        if exercise_type != "american":
            raise ValueError("Exercise type must be 'European' or 'American'")
        payoff = self.tree.option.intrinsic_value(self.spot_price)
        # Une égalité à l'arrondi près ne déclenche pas d'exercice.
        # Cette convention évite des décisions artificielles sur un
        # call sans dividende ou sur des nœuds de payoff nul.
        tolerance = self.tree.exercise_tolerance(payoff, self.continuation_value)
        self.exercise_now = payoff > 0 and payoff > self.continuation_value + tolerance
        # Dans la bande d'égalité numérique on conserve la continuation.
        self.option_value = payoff if self.exercise_now else self.continuation_value


# -----------------------------------------------------------------------------
# TRUNK NODE : nœud du tronc, point d'entrée de chaque colonne.
# -----------------------------------------------------------------------------
class TrunkNode(Node):
    """Nœud du tronc : relie aussi les dates entre elles."""

    def __init__(self, tree: Tree, spot_price: float, step: int) -> None:
        # Initialiser d'abord tous les champs d'un nœud ordinaire.
        super().__init__(tree, spot_price, step)
        # Liens supplémentaires propres au tronc.
        self.previous_trunk: TrunkNode | None = None
        self.next_trunk: TrunkNode | None = None
        self.column_size = 1

    # Vrai pour la racine (pas de colonne précédente)
    def _is_root(self) -> bool:
        """Vrai pour la racine (pas de colonne précédente)."""
        return self.previous_trunk is None


# -----------------------------------------------------------------------------
# TREE : construction, pricing et profils.
# -----------------------------------------------------------------------------
class Tree:
    """Graphe recombiné du sous-jacent, indépendant du strike et du payoff.

    1. ``build`` avance dans le temps et additionne les probabilités entrantes.
    2. ``price`` remonte le temps pour valoriser le contrat.
    3. ``exercise_profile`` avance à nouveau pour les statistiques d'arrêt.
    """

    def __init__(self, model: Model, option: Option, market: Market) -> None:
        self.model = model
        self.option = option
        self.market = market
        self.duration_days = (option.maturity - model.pricing_date).days
        if self.duration_days <= 0:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        # Dérive du sous-jacent : le rendement q réduit la croissance.
        self.drift = market.rate - market.dividend_yield
        self._locate_dividend()
        self._set_step_parameters()
        # Racine : spot initial, atteinte avec probabilité 1.
        self.root = TrunkNode(tree=self, spot_price=market.spot, step=0)
        self.root.proba = 1.0
        self.last_trunk = self.root
        self._built = False

    # Affecter l'unique dividende cash à l'intervalle (t_i, t_{i+1}]
    def _locate_dividend(self) -> None:
        """Affecter l'unique dividende cash à l'intervalle (t_i, t_{i+1}]."""
        self.dividend_step: int | None = None
        self.remaining_after_dividend = 0.0
        market, model = self.market, self.model
        # Un cash hors de la vie du contrat n'a aucun effet sur l'arbre.
        if not (market.dividend_amount > 0
                and model.pricing_date < market.ex_dividend_date <= self.option.maturity):
            return
        # Calcul entier : évite les erreurs d'arrondi de dates signalées dans le cours.
        ex_days = (market.ex_dividend_date - model.pricing_date).days
        n, days = model.nb_steps, self.duration_days
        self.dividend_step = (ex_days * n + days - 1) // days
        # Temps (en années) entre le détachement et la fin du pas.
        self.remaining_after_dividend = (self.dividend_step * days - ex_days * n) / (365.0 * n)

    # Pas de temps, actualisation, croissance et probabilités binomiales
    def _set_step_parameters(self) -> None:
        """Pas de temps, actualisation, croissance et probabilités binomiales."""
        maturity_years = self.duration_days / 365.0
        self.deltaT = maturity_years / self.model.nb_steps
        # Facteur d'actualisation sur un pas, et croissance du forward.
        self.df = np.exp(-self.market.rate * self.deltaT)
        self.growth = np.exp(self.drift * self.deltaT)
        self.variance_factor = np.expm1(self.market.vol**2 * self.deltaT)
        self.sqrtAlpha = self._compute_sqrt_alpha()
        # Ratio entre deux voisins d'une colonne, y compris le pas trinomial.
        self.alpha = self.sqrtAlpha**2
        if self.market.vol > 0 and (not np.isfinite(self.alpha) or self.alpha <= 1):
            raise ValueError("La volatilité et le pas de temps produisent "
                             "un maillage non représentable.")
        # Probabilités binomiales communes (moyenne exacte autour du forward).
        self.p_up = 1 / (1 + self.sqrtAlpha)
        self.p_down = self.sqrtAlpha / (1 + self.sqrtAlpha)

    # Facteur binomial >1 obtenu en égalant moyenne et variance (p7–8)
    def _compute_sqrt_alpha(self) -> float:
        """Facteur binomial >1 obtenu en égalant moyenne et variance (p7–8).

        Le nom historique ``sqrtAlpha`` est conservé. Il correspond au
        facteur alpha BINOMIAL des diapositives ; ``self.alpha`` est son
        carré, c'est-à-dire le rapport entre voisins d'une même colonne.
        ``expm1(x)`` évite la perte de précision de ``exp(x)-1`` pour x petit.
        """
        a = 1 + self.variance_factor / 2
        # Cette formule donne directement sqrt(alpha).
        return a + np.sqrt(self.variance_factor * (1 + self.variance_factor / 4))

    # ------------------------------------------------------------------
    # Moments du sous-jacent sur un pas (utilisés par Node).
    # ------------------------------------------------------------------
    def transition_type(self, step: int) -> str:
        """Type de la transition step -> step+1 (binomiale, trinomiale...)."""
        if step >= self.model.nb_steps:
            return "terminal"
        if self.market.vol == 0:
            return "deterministic"
        # Le pas qui contient le détachement cash est trinomial.
        return "trinomial" if step + 1 == self.dividend_step else "binomial"

    # Forward d'un pas, diminué du cash capitalisé s'il est détaché
    def expected_spot(self, spot: float, step: int) -> float:
        """Forward d'un pas, diminué du cash capitalisé s'il est détaché."""
        mean = spot * self.growth
        if step + 1 == self.dividend_step:
            mean -= self.market.dividend_amount * np.exp(
                self.drift * self.remaining_after_dividend)
        return float(mean)

    # Variance d'un pas ; approximation intra-pas au détachement cash
    def conditional_variance(self, spot: float, step: int) -> float:
        """Variance d'un pas ; approximation intra-pas au détachement cash."""
        base = spot
        if step + 1 == self.dividend_step:
            # Approximation intra-pas du cours: w = (t_fin-t_ex)/dt.
            # Ce n'est pas la variance exacte d'un GBM avec saut cash intérieur.
            base -= self.market.dividend_amount * self.remaining_after_dividend / self.deltaT
        return float((base * self.growth)**2 * self.variance_factor)

    # ------------------------------------------------------------------
    # Parcours par liens directs (aucune liste de nœuds).
    # ------------------------------------------------------------------
    def iter_columns(self) -> Iterator[TrunkNode]:
        """Avancer le long du tronc avec next_trunk, date par date."""
        trunk = self.root
        while trunk is not None:
            yield trunk
            trunk = trunk.next_trunk

    # Monter en haut d'une colonne par above, puis redescendre par below
    def iter_column(self, trunk: TrunkNode) -> Iterator[Node]:
        """Monter en haut d'une colonne par above, puis redescendre par below."""
        node: Node = trunk
        while node.above is not None:
            node = node.above
        while node is not None:
            yield node
            node = node.below

    # Nombre total de nœuds de l'arbre (pour mesurer l'effet du pruning)
    def count_nodes(self) -> int:
        """Nombre total de nœuds de l'arbre (pour mesurer l'effet du pruning)."""
        return sum(trunk.column_size for trunk in self.iter_columns())

    # Appliquer une fonction au tronc, puis au-dessus, puis en dessous
    def _apply_to_column(self, trunk: TrunkNode, function: Callable[[Node], None]) -> None:
        """Appliquer une fonction au tronc, puis au-dessus, puis en dessous."""
        function(trunk)
        # Nœuds au-dessus du tronc.
        node = trunk.above
        while node is not None:
            function(node)
            node = node.above
        # Nœuds en dessous du tronc.
        node = trunk.below
        while node is not None:
            function(node)
            node = node.below

    # ------------------------------------------------------------------
    # Construction : outils communs.
    # ------------------------------------------------------------------
    def _positive_mean(self, node: Node) -> float:
        """Forward du nœud, refusé s'il n'est pas strictement positif."""
        mean = node.forward()
        if mean <= 0 or not np.isfinite(mean):
            raise ValueError(
                f"Pas {node.step} -> {node.step+1}, S={node.spot_price:.10g} : "
                "espérance après dividende non positive. Le maillage cash positif ne "
                "peut représenter ce nœud. Réduire N pour un essai ou étudier un "
                "élagage explicite ; le dividende n'est pas tronqué silencieusement."
            )
        return mean

    # Créer un nœud ordinaire (hors tronc)
    def _create_node(self, spot_price: float, step: int) -> Node:
        """Créer un nœud ordinaire (hors tronc)."""
        return Node(tree=self, spot_price=spot_price, step=step)

    # Connecter deux voisins de la même colonne
    @staticmethod
    def _connect_vertical(upper_node: Node, lower_node: Node) -> None:
        """Connecter deux voisins de la même colonne."""
        upper_node.below = lower_node
        lower_node.above = upper_node

    # Voisin du dessus, créé si besoin (on étend la chaîne, pas un tableau)
    def _upper_neighbor(self, node: Node, trunk: TrunkNode) -> Node:
        """Voisin du dessus, créé si besoin (on étend la chaîne, pas un tableau)."""
        if node.above is None:
            upper = self._create_node(node.spot_price * self.alpha, node.step)
            self._connect_vertical(upper, node)
            trunk.column_size += 1
        return node.above

    # Voisin du dessous, créé si besoin
    def _lower_neighbor(self, node: Node, trunk: TrunkNode) -> Node:
        """Voisin du dessous, créé si besoin."""
        if node.below is None:
            lower = self._create_node(node.spot_price / self.alpha, node.step)
            self._connect_vertical(node, lower)
            trunk.column_size += 1
        return node.below

    # ------------------------------------------------------------------
    # Construction : pas trinomial du dividende cash.
    # ------------------------------------------------------------------
    def _closest_middle(self, candidate: Node, mean: float, trunk: TrunkNode) -> Node:
        """Nœud de la colonne le plus proche (en prix) du forward ``mean``."""
        # On monte tant que le voisin du dessus est plus proche.
        while mean > candidate.spot_price:
            upper = self._upper_neighbor(candidate, trunk)
            if abs(upper.spot_price - mean) >= abs(candidate.spot_price - mean):
                break
            candidate = upper
        # Puis on descend tant que le voisin du dessous est plus proche.
        while mean < candidate.spot_price:
            lower = self._lower_neighbor(candidate, trunk)
            if abs(lower.spot_price - mean) >= abs(candidate.spot_price - mean):
                break
            candidate = lower
        return candidate

    # Résoudre les 3 moments centrés (plus stable que E²+V puis soustraction)
    def _trinomial_probabilities(self, node: Node, middle: Node) -> tuple[float, float, float]:
        """Résoudre les 3 moments centrés (plus stable que E²+V puis soustraction)."""
        mean = self._positive_mean(node)
        u = node.next_up.spot_price - middle.spot_price
        d = node.next_down.spot_price - middle.spot_price
        e = mean - middle.spot_price
        second = node.variance() + e * e
        p_up = (second - e * d) / (u * (u - d))
        p_down = (second - e * u) / (d * (d - u))
        return p_up, 1.0 - p_up - p_down, p_down

    # Refuser des probabilités négatives plutôt que de les forcer
    @staticmethod
    def _check_trinomial(node: Node, probabilities: tuple[float, float, float]) -> None:
        """Refuser des probabilités négatives plutôt que de les forcer."""
        tolerance = 1e-12
        if all(np.isfinite(p) and -tolerance <= p <= 1 + tolerance for p in probabilities):
            return
        p_up, p_mid, p_down = probabilities
        raise ValueError(
            f"Pas {node.step} -> {node.step+1}, S={node.spot_price:.10g} : "
            f"probabilités trinômiales invalides (u={p_up:.6g}, m={p_mid:.6g}, "
            f"d={p_down:.6g}). Les moments du cours ne sont pas représentables sur "
            "ces trois nœuds ; modifier le maillage ou étudier un élagage, "
            "sans forcer les probabilités."
        )

    # Relier un parent à trois enfants voisins et fixer leurs probabilités
    def _connect_trinomial(self, node: Node, middle: Node, trunk: TrunkNode) -> None:
        """Relier un parent à trois enfants voisins et fixer leurs probabilités."""
        node.next_mid = middle
        node.next_up = self._upper_neighbor(middle, trunk)
        node.next_down = self._lower_neighbor(middle, trunk)
        probabilities = self._trinomial_probabilities(node, middle)
        self._check_trinomial(node, probabilities)
        p_up, p_mid, p_down = probabilities
        # Ne corriger que d'éventuels débordements dus à l'arrondi machine.
        node.p_up = float(min(1.0, max(0.0, p_up)))
        node.p_down = float(min(1.0, max(0.0, p_down)))
        node.p_mid = float(min(1.0, max(0.0, p_mid)))
        total = node.p_up + node.p_mid + node.p_down
        node.p_up /= total
        node.p_mid /= total
        node.p_down /= total

    # Pas cash : tronc au forward puis trois enfants voisins par parent
    def _build_dividend_column(self, trunk: TrunkNode, step: int) -> TrunkNode:
        """Pas cash : tronc au forward puis trois enfants voisins par parent.

        Au-dessus et au-dessous, le curseur ``candidate`` reste sur la
        nouvelle colonne. Il se déplace par les liens verticaux pour trouver
        le milieu le plus proche du forward de chaque parent. Les voisins
        déjà créés sont réutilisés : l'arbre reste recombiné. Ce pas n'est
        volontairement pas élagué dans la version fournie.
        """
        next_trunk = TrunkNode(self, self._positive_mean(trunk), step)
        self._connect_trinomial(trunk, next_trunk, next_trunk)
        # Même traitement vers le haut (above) puis vers le bas (below).
        for direction in ("above", "below"):
            current = getattr(trunk, direction)
            candidate: Node = next_trunk
            while current is not None:
                mean = self._positive_mean(current)
                candidate = self._closest_middle(candidate, mean, next_trunk)
                self._connect_trinomial(current, candidate, next_trunk)
                current = getattr(current, direction)
        return next_trunk

    # ------------------------------------------------------------------
    # Construction : pas binomial (tronc, partie haute, partie basse).
    # ------------------------------------------------------------------
    def _build_trunk_up(self, trunk: TrunkNode, step: int) -> TrunkNode:
        """Créer les deux enfants du tronc lorsque le tronc monte."""
        fwd = trunk.forward()
        next_trunk = TrunkNode(self, fwd * self.sqrtAlpha, step)
        lower_node = self._create_node(fwd / self.sqrtAlpha, step)
        trunk.next_up = next_trunk
        trunk.next_down = lower_node
        self._connect_vertical(next_trunk, lower_node)
        return next_trunk

    # Créer les deux enfants du tronc lorsque le tronc descend
    def _build_trunk_down(self, trunk: TrunkNode, step: int) -> TrunkNode:
        """Créer les deux enfants du tronc lorsque le tronc descend."""
        fwd = trunk.forward()
        next_trunk = TrunkNode(self, fwd / self.sqrtAlpha, step)
        upper_node = self._create_node(fwd * self.sqrtAlpha, step)
        trunk.next_down = next_trunk
        trunk.next_up = upper_node
        self._connect_vertical(upper_node, next_trunk)
        return next_trunk

    # Pruning : seul un nœud du bord peut être coupé, s'il est trop improbable
    def _should_prune(self, node: Node, is_edge: bool) -> bool:
        """Pruning : seul un nœud du bord peut être coupé, s'il est trop improbable."""
        return is_edge and node.proba < self.model.pruning_threshold

    # Un nœud coupé va vers un seul enfant (déjà existant), avec probabilité 1
    @staticmethod
    def _connect_monomial(node: Node, child: Node) -> None:
        """Un nœud coupé va vers un seul enfant (déjà existant), avec probabilité 1."""
        node.is_pruned = True
        node.next_up = None
        node.next_down = None
        node.next_mid = child
        node.p_up, node.p_mid, node.p_down = 0.0, 1.0, 0.0

    # Construire au-dessus du tronc ; renvoie 1 si le bord haut est coupé
    def _build_above(self, trunk: TrunkNode, step: int) -> int:
        """Construire au-dessus du tronc ; renvoie 1 si le bord haut est coupé."""
        current = trunk.above
        while current is not None:
            # L'enfant bas est l'enfant haut du voisin du dessous.
            down_child = current.below.next_up
            if self._should_prune(current, current.above is None):
                # Pas de nouvel enfant en haut : la colonne ne grandit pas.
                self._connect_monomial(current, down_child)
                return 1
            current.next_down = down_child
            new_up = self._create_node(current.forward() * self.sqrtAlpha, step)
            current.next_up = new_up
            self._connect_vertical(new_up, current.next_down)
            current = current.above
        return 0

    # Construire sous le tronc ; renvoie 1 si le bord bas est coupé
    def _build_below(self, trunk: TrunkNode, step: int) -> int:
        """Construire sous le tronc ; renvoie 1 si le bord bas est coupé."""
        current = trunk.below
        while current is not None:
            # L'enfant haut est l'enfant bas du voisin du dessus.
            up_child = current.above.next_down
            if self._should_prune(current, current.below is None):
                self._connect_monomial(current, up_child)
                return 1
            current.next_up = up_child
            new_down = self._create_node(current.forward() / self.sqrtAlpha, step)
            current.next_down = new_down
            self._connect_vertical(current.next_up, new_down)
            current = current.below
        return 0

    # Ajouter proba(parent) × p(branche) à chaque enfant existant
    @staticmethod
    def _push_probability(node: Node) -> None:
        """Ajouter proba(parent) × p(branche) à chaque enfant existant."""
        # Il faut ``+=`` : un enfant recombiné reçoit plusieurs chemins.
        if node.next_up is not None:
            node.next_up.proba += node.p_up * node.proba
        if node.next_mid is not None:
            node.next_mid.proba += node.p_mid * node.proba
        if node.next_down is not None:
            node.next_down.proba += node.p_down * node.proba

    # Transmettre la probabilité de chaque nœud de la colonne à ses enfants
    def _propagate_probabilities(self, trunk: TrunkNode) -> None:
        """Transmettre la probabilité de chaque nœud de la colonne à ses enfants.

        La colonne suivante est neuve et ses probabilités commencent à zéro.
        Ces masses décrivent le sous-jacent, sans arrêt américain ; elles
        servent aussi à décider l'élagage à la prochaine construction.
        """
        self._apply_to_column(trunk, self._push_probability)

    # Volatilité nulle : un seul enfant, au forward, avec probabilité 1
    def _build_deterministic(self, trunk: TrunkNode, step: int) -> TrunkNode:
        """Volatilité nulle : un seul enfant, au forward, avec probabilité 1."""
        next_trunk = TrunkNode(self, self._positive_mean(trunk), step)
        trunk.next_mid = next_trunk
        trunk.p_up, trunk.p_mid, trunk.p_down = 0.0, 1.0, 0.0
        return next_trunk

    # Construire une date puis la relier aux deux dates voisines
    def _build_next_column(self, trunk: TrunkNode, step: int) -> TrunkNode:
        """Construire une date puis la relier aux deux dates voisines.

        Hors cash, le tronc monte aux pas impairs et descend aux pas pairs.
        Il reste près du centre du triangle, permettant deux parcours
        indépendants : vers le haut puis vers le bas. La formule de taille
        ``ancienne + 1 - bords_coupés`` compte les objets réellement créés.
        """
        kind = self.transition_type(step - 1)
        # Choisir comment créer le prochain tronc selon le type de pas.
        if kind == "deterministic":
            next_trunk = self._build_deterministic(trunk, step)
        elif kind == "trinomial":
            next_trunk = self._build_dividend_column(trunk, step)
        elif step % 2 == 1:
            next_trunk = self._build_trunk_up(trunk, step)
        else:
            next_trunk = self._build_trunk_down(trunk, step)
        # Relier les dates pour pouvoir ensuite remonter l'arbre.
        trunk.next_trunk = next_trunk
        next_trunk.previous_trunk = trunk
        if kind == "binomial":
            pruned = self._build_above(trunk, step) + self._build_below(trunk, step)
            # Une colonne binomiale a un nœud de plus, moins les bords coupés.
            next_trunk.column_size = trunk.column_size + 1 - pruned
        self._propagate_probabilities(trunk)
        return next_trunk

    # Avancer de la racine à la dernière colonne (idempotent)
    def build(self) -> Tree:
        """Avancer de la racine à la dernière colonne (idempotent)."""
        if self._built:
            return self
        if self.root.next_trunk is not None:
            raise RuntimeError("Construction précédente incomplète : créer un nouvel arbre.")
        trunk = self.root
        # Une colonne par pas de temps, de t1 jusqu'à la maturité.
        for step in range(1, self.model.nb_steps + 1):
            trunk = self._build_next_column(trunk, step)
        self.last_trunk = trunk
        self._built = True
        return self

    # Afficher chaque colonne dans le terminal (petits arbres seulement)
    def print_tree(self) -> None:
        """Afficher chaque colonne dans le terminal (petits arbres seulement)."""
        for trunk in self.iter_columns():
            print(f"\nSTEP {trunk.step}")
            # On parcourt la colonne de haut en bas.
            for node in self.iter_column(trunk):
                trunk_marker = " <-- TRUNK" if node is trunk else ""
                print(f"spot = {node.spot_price:.4f}{trunk_marker}")
                if node.option_value is not None:
                    print(f"option = {node.option_value:.6f}")

    # -------------------------------------------------------------------------
    # VALEUR ET EXERCICE : mouvement arrière, puis mouvement avant des masses.
    # -------------------------------------------------------------------------
    @staticmethod
    def exercise_tolerance(payoff: float, continuation: float) -> float:
        """Tolérance absolue + relative, exprimée dans l'unité du payoff."""
        return 1e-12 * max(1.0, abs(payoff), abs(continuation))

    # À maturité : valeur = payoff ; exercice si payoff strictement positif
    def _set_terminal_values(self) -> None:
        """À maturité : valeur = payoff ; exercice si payoff strictement positif."""
        for node in self.iter_column(self.last_trunk):
            payoff = self.option.intrinsic_value(node.spot_price)
            node.option_value = payoff
            node.continuation_value = None
            # Une option à la monnaie ou hors monnaie expire sans exercice.
            node.exercise_now = payoff > 0.0

    # Recalculer toutes les valeurs et décisions pour l'option actuelle
    def price(self) -> float:
        """Recalculer toutes les valeurs et décisions pour l'option actuelle.

        On peut changer le strike, call/put ou européen/américain puis appeler
        cette méthode sans reconstruire le sous-jacent. Modifier le marché,
        les dates ou N nécessite en revanche un nouvel objet ``Tree``.
        """
        if not self._built:
            raise RuntimeError("Appeler tree.build() avant tree.price().")
        self._set_terminal_values()
        # Remonter colonne par colonne jusqu'à la racine.
        trunk = self.last_trunk.previous_trunk
        while trunk is not None:
            for node in self.iter_column(trunk):
                node.compute_option_value()
            trunk = trunk.previous_trunk
        return float(self.root.option_value)

    # Espérance théorique de S(t), dividende cash déjà détaché inclus
    def mean_at_time(self, time_years: float) -> float:
        """Espérance théorique de S(t), dividende cash déjà détaché inclus.

        Cette référence concerne le modèle avant élagage. Le pruning fourni
        redirige les branches des bords et peut donc la modifier légèrement.
        """
        maturity_years = self.duration_days / 365.0
        if not np.isfinite(time_years) or not 0 <= time_years <= maturity_years + 1e-12:
            raise ValueError("Le temps doit être compris entre 0 et la maturité.")
        mean = self.market.spot * np.exp(self.drift * time_years)
        # Retirer le cash capitalisé une fois sa date passée.
        if self.dividend_step is not None:
            ex_time = (self.market.ex_dividend_date - self.model.pricing_date).days / 365.0
            if time_years >= ex_time - 1e-12:
                mean -= self.market.dividend_amount * np.exp(self.drift * (time_years - ex_time))
        return float(mean)

    # Call européen K=0 théorique = forward final actualisé (cours p21)
    def expected_zero_strike(self) -> float:
        """Call européen K=0 théorique = forward final actualisé (cours p21)."""
        maturity_years = self.duration_days / 365.0
        return float(np.exp(-self.market.rate * maturity_years)
                     * self.mean_at_time(maturity_years))

    # Remettre à zéro les masses vivantes ; la racine repart à 1
    def _reset_alive_masses(self) -> None:
        """Remettre à zéro les masses vivantes ; la racine repart à 1."""
        for trunk in self.iter_columns():
            for node in self.iter_column(trunk):
                node.alive_proba = 0.0
        self.root.alive_proba = 1.0

    # Sommer les masses d'une colonne et propager les masses vivantes
    def _scan_column(self, trunk: TrunkNode) -> tuple[float, float, float, float, float, float]:
        """Sommer les masses d'une colonne et propager les masses vivantes.

        Renvoie (moyenne, zone d'exercice, premier exercice, valeur vivante,
        flux payé, survie), sans actualisation.
        """
        mean_value = region = first = alive_value = cashflow = survival = 0.0
        for node in self.iter_column(trunk):
            # La moyenne brute peut contenir des trajectoires qui auraient
            # déjà exercé. C'est volontaire pour reproduire le PDF.
            mean_value += node.proba * node.option_value
            alive_value += node.alive_proba * node.option_value
            if node.exercise_now:
                region += node.proba
                first += node.alive_proba
                cashflow += node.alive_proba * self.option.intrinsic_value(node.spot_price)
                # Absorption : aucune masse de ce nœud ne va aux enfants.
                continue
            survival += node.alive_proba
            for branch in BRANCHES:
                child = getattr(node, "next_" + branch)
                if child is not None:
                    child.alive_proba += node.alive_proba * getattr(node, "p_" + branch)
        return mean_value, region, first, alive_value, cashflow, survival

    # Calculer les profils de valeur et d'exercice du cours, pages 22–23
    def exercise_profile(self) -> tuple[ExerciseProfileRow, ...]:
        """Calculer les profils de valeur et d'exercice du cours, pages 22–23.

        Premier passage (``price``) : résolution arrière du problème d'arrêt.
        Second passage : propagation avant de deux distributions distinctes.

        * ``proba`` existe depuis la construction : on ignore les exercices.
          Sa moyenne actualisée de V est la courbe demandée dans le PDF.
        * ``alive_proba`` est absorbée au PREMIER exercice. Elle permet de
          calculer une vraie distribution des dates de paiement et son cumul.

        Il n'y a ni renormalisation ni actualisation des probabilités.
        """
        self.price()  # Évite de réutiliser les décisions d'un ancien strike.
        self._reset_alive_masses()
        rows: list[ExerciseProfileRow] = []
        cumulative = past_cashflows = 0.0
        start = dt.datetime.combine(self.model.pricing_date, dt.time())
        # Une ligne par date, de la racine à la maturité.
        for trunk in self.iter_columns():
            mean, region, first, alive, cash, survival = self._scan_column(trunk)
            time_years = trunk.step * self.deltaT
            discount = float(np.exp(-self.market.rate * time_years))
            cumulative += first
            rows.append(ExerciseProfileRow(
                step=trunk.step, time_years=time_years,
                date=start + dt.timedelta(
                    days=trunk.step * self.duration_days / self.model.nb_steps),
                discounted_mean_value=discount * mean,
                exercise_region_probability=region, first_exercise_probability=first,
                cumulative_exercise_probability=cumulative, survival_probability=survival,
                discounted_alive_value=discount * alive,
                discounted_exercise_cashflow=discount * cash,
                cumulative_discounted_cashflow=past_cashflows + discount * cash,
                # Valeur vivante AVANT décision + flux STRICTEMENT antérieurs,
                # sinon le paiement du jour serait compté deux fois.
                total_value=discount * alive + past_cashflows,
            ))
            past_cashflows += discount * cash
        return tuple(rows)


# -----------------------------------------------------------------------------
# MAIN : exemple lancé directement depuis VS Code (sans Streamlit).
# -----------------------------------------------------------------------------
def _print_report(tree: Tree, price: float, elapsed: float) -> None:
    """Afficher les paramètres de l'exemple et le prix obtenu."""
    market, option, model = tree.market, tree.option, tree.model
    print("ARBRE BINOMIAL / PAS TRINOMIAL SI DIVIDENDE CASH")
    print(f"Date de calcul : {model.pricing_date:%d/%m/%Y}")
    print(f"Maturité : {option.maturity:%d/%m/%Y}")
    print(f"Option : {option.type} {option.exercise_type}")
    print(f"Spot : {market.spot} | Strike : {option.strike}")
    print(f"Taux : {market.rate:.1%} | Volatilité : {market.vol:.1%}")
    print(f"Rendement continu q : {market.dividend_yield:.1%}")
    print(f"Dividende cash : {market.dividend_amount:g} | Ex-date : {market.ex_dividend_date}")
    print(f"Nombre de pas : {model.nb_steps}")
    print(f"Seuil de pruning : {model.pruning_threshold:g}")
    print(f"\nPrix de l'option : {price:.10f}")
    print(f"Nombre de nœuds : {tree.count_nodes()}")
    print(f"Temps de calcul : {elapsed:.2f} s")


# Pricer un put européen avec dividende cash et afficher le résultat
def main() -> None:
    """Pricer un put européen avec dividende cash et afficher le résultat."""
    # pruning_threshold = 0 : pas de pruning ; 1e-7 : on coupe les nœuds de proba < 1e-7
    model = Model(pricing_date=dt.date(2026, 9, 15), nb_steps=1000, pruning_threshold=1e-7)
    option = Option(strike=101, maturity=dt.date(2027, 9, 2),
                    type="Put", exercise_type="European")
    market = Market(spot=100, rate=0.03, vol=0.20, dividend_yield=0,
                    dividend_amount=3, ex_dividend_date=dt.date(2027, 3, 15))
    # Toujours construire l'arbre avant de calculer le prix (on chronomètre les deux).
    start = perf_counter()
    tree = Tree(model=model, option=option, market=market).build()
    price = tree.price()
    _print_report(tree, price, perf_counter() - start)
    # tree.print_tree() # Permet d'afficher chaque step.


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# PRUNING SUR LA PARTIE BINOMIALE
# -----------------------------------------------------------------------------
# Le cours présente le pruning sur un arbre trinomial : un nœud trop improbable
# ne branche plus que vers son nœud du milieu, avec probabilité 1. Dans notre
# arbre binomial, il n'y a pas de nœud du milieu, donc on l'a adapté ainsi :
#
# 1. Chaque nœud garde sa probabilité d'être atteint depuis la racine (proba).
#    La racine vaut 1, puis chaque nœud transmet proba × p_up et proba × p_down
#    à ses enfants (_propagate_probabilities).
#
# 2. Seuls les nœuds du bord (tout en haut ou tout en bas d'une colonne)
#    peuvent être coupés : cela évite de créer des trous dans une colonne.
#    Le seuil porte sur leur masse totale, y compris les contributions déjà
#    redirigées par des coupes antérieures.
#
# 3. Si un nœud du bord a une proba < pruning_threshold, on ne crée pas de
#    nouvel enfant vers l'extérieur. Il va, avec probabilité 1, vers l'enfant
#    déjà créé par son voisin (_connect_monomial dans _build_above/_build_below).
#    La colonne suivante ne grandit donc pas de ce côté à ce pas.
#
# Limite : le nœud redirigé ne va pas exactement vers son forward, donc
# l'espérance et la variance locales sont modifiées. Un seuil sur la seule
# probabilité ne donne PAS de borne générale d'erreur de prix : le payoff et
# le nombre de coupes comptent aussi. Le pas trinomial du dividende cash n'est
# pas élagué : il est construit en entier, comme sans pruning.
