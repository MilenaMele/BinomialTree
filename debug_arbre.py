"""Débogueur de l'arbre : contrôles numériques et dessin SVG.

À quoi sert ce module ?
    Il vérifie qu'un arbre construit par ``arbre_binomial.py`` est cohérent
    (liens, probabilités, moments, valeurs de l'option) et le dessine en
    SVG pour pouvoir l'inspecter dans un navigateur. La page « Arbre &
    diagnostics » de l'interface Streamlit l'utilise.

Deux usages complémentaires :

* ``DebugTree(tree).check()`` parcourt tous les liens et renvoie un nombre
  d'erreurs. Il convient aussi aux grands arbres, sans fenêtre graphique.
* ``DebugTree(tree).export_svg("arbre.svg")`` produit un dessin vectoriel :
  on peut ouvrir le fichier dans un navigateur et zoomer sans pixellisation.

Comme le demande le cours (p. 18), aucun tableau, dictionnaire ou liste de
nœuds ne sert à accéder à l'arbre : on suit uniquement les liens du tronc et
les voisins ``above``/``below``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Iterator, TextIO
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_hex

from arbre_binomial import BRANCHES, Market, Model, Node, Option, Tree, TrunkNode


# Limiter le nombre de nœuds plutôt que le nombre de dates permet de montrer
# un pruning sur 20 pas, tout en évitant de produire un SVG de plusieurs Mo
# par inadvertance. L'appelant peut augmenter cette limite explicitement.
DEFAULT_MAX_DRAW_NODES = 1_200

# Feuille de style et éléments fixes du SVG (texte constant, sans nœud).
_SVG_DEFS = (
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
    'markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
    'fill="context-stroke"/></marker></defs>\n'
)
_SVG_STYLE = (
    '<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:12px;fill:#172033}'
    '.heading{font-size:23px;font-weight:700}.subheading{font-size:14px}'
    '.small{font-size:11px}.decision{font-size:10px;font-weight:700}'
    '.edge-label{font-size:10px;paint-order:stroke;stroke:#fff;stroke-width:4px;'
    'stroke-linejoin:round}.forward{fill:none;stroke:#0284c7;stroke-width:2;'
    'stroke-dasharray:7 5}</style>\n'
)
_SVG_DESCRIPTION = (
    '<desc id="description">Dessin vectoriel. Axe vertical logarithmique. '
    "S : prix du sous-jacent ; V : valeur de l’option ; π : probabilité du nœud ; "
    "p : probabilité conditionnelle de branche.</desc>\n"
)
# Lignes d'aide fixes affichées sous le titre du dessin.
_LEGEND_LINES = (
    (65, "S : sous-jacent (€) · V : option (€) · π : probabilité du nœud · "
         "p : probabilité conditionnelle de branche", "subheading"),
    (89, "Contours : rouge = exercice anticipé ; orange = pruning ; double = tronc. "
         "Flèche orange p=1 : masse redirigée vers l’intérieur.", "small"),
    (111, "π ne tient pas compte des exercices antérieurs. Axe vertical : log(S). "
          "Pointillés bleus : forward théorique sans biais de pruning.", "small"),
)
# Dimensions des boîtes et des marges, en pixels.
_BOX_WIDTH, _BOX_HEIGHT = 174.0, 76.0
_MARGIN_X, _TOP_Y, _STEP_WIDTH = 130.0, 230.0, 270.0


class _SvgCanvas:
    """Coordonnées d'un nœud sur le dessin et écriture des textes SVG."""

    def __init__(self, stream: TextIO, max_log: float, scale_y: float) -> None:
        # Le haut du dessin correspond au plus grand log(S).
        self.stream = stream
        self.max_log = max_log
        self.scale_y = scale_y

    def x(self, node: Node) -> float:
        """Abscisse : une colonne par pas de temps."""
        return _MARGIN_X + node.step * _STEP_WIDTH

    def y_spot(self, spot: float) -> float:
        """Ordonnée logarithmique : deux voisins sont également espacés."""
        return _TOP_Y + (self.max_log - math.log(spot)) * self.scale_y

    def y(self, node: Node) -> float:
        """Ordonnée du nœud à partir de son prix."""
        return self.y_spot(node.spot_price)

    @staticmethod
    def _text_markup(px: float, py: float, label: str, css: str, anchor: str) -> str:
        """Balise <text> avec un libellé échappé (< > & sont protégés)."""
        return (f'<text x="{px:.2f}" y="{py:.2f}" text-anchor="{anchor}" '
                f'class="{css}">{escape(label)}</text>\n')

    def text(self, px: float, py: float, label: str, css: str = "",
             anchor: str = "middle") -> None:
        """Écrire un texte positionné sur le dessin."""
        self.stream.write(self._text_markup(px, py, label, css, anchor))

    def write(self, markup: str) -> None:
        """Écriture brute d'un élément déjà formaté."""
        self.stream.write(markup)


class DebugTree:
    """Contrôler structure, probabilités, moments et valeurs de l'option.

    Trois curseurs de parents suivent respectivement les branches up, mid et
    down, dans l'ordre décroissant des spots. Quand on descend dans la colonne
    des enfants, ces curseurs ne remontent jamais : les probabilités entrantes
    sont vérifiées en temps linéaire. Un nœud élagué envoie toute sa masse
    vers un enfant intérieur : ce biais de moments est mesuré et annoncé ; il
    ne constitue pas une erreur de construction.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self._reset_counters()
        # Curseurs de recherche (enfants puis parents) pour chaque branche.
        self._child_up: Node | None = None
        self._child_mid: Node | None = None
        self._child_down: Node | None = None
        self._parent_up: Node | None = None
        self._parent_mid: Node | None = None
        self._parent_down: Node | None = None

    def _reset_counters(self) -> None:
        """Remettre à zéro les compteurs et les biais avant un contrôle."""
        self.errors = self.node_count = self.branch_count = 0
        self.pruned_node_count = 0
        # Biais de moyenne/variance introduits par les coupes du pruning.
        self.pruning_weighted_mean_bias = 0.0
        self.pruning_weighted_abs_mean_bias = 0.0
        self.pruning_weighted_variance_bias = 0.0
        self.pruning_max_abs_mean_bias = 0.0
        self.terminal_spot_expectation = 0.0
        self.discounted_terminal_mean_bias = 0.0
        self._previous_has_pruning = False

    # ------------------------------------------------------------------
    # Outils de base : tolérance, erreurs et parcours sécurisés.
    # ------------------------------------------------------------------
    @staticmethod
    def _close(actual: float, expected: float) -> bool:
        """Tolérance pour les sommes et moments, sans exiger l'égalité binaire."""
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-10))

    def _error(self, message: str, node: Node | None = None) -> None:
        """Compter une erreur et l'afficher avec la position du nœud."""
        self.errors += 1
        location = "" if node is None else f"Pas {node.step}, S={node.spot_price:.6f} : "
        print("ERREUR :", location + message)

    def _columns(self) -> Iterator[TrunkNode]:
        """Parcours borné : un cycle du tronc ne peut pas boucler indéfiniment."""
        trunk = self.tree.root
        previous = None
        for step in range(self.tree.model.nb_steps + 1):
            # Chaque tronc doit être à la bonne date et relié au précédent.
            if not isinstance(trunk, TrunkNode) or trunk.step != step:
                raise ValueError(f"Tronc absent ou mauvaise date au pas {step}")
            if trunk.previous_trunk is not previous:
                raise ValueError(f"Lien précédent du tronc incorrect au pas {step}")
            yield trunk
            previous, trunk = trunk, trunk.next_trunk
        if trunk is not None or previous is not self.tree.last_trunk:
            raise ValueError("Fin du tronc incorrecte")

    @staticmethod
    def _top(trunk: TrunkNode) -> Node:
        """Sommet de la colonne, avec détection d'un cycle de liens above."""
        node: Node = trunk
        moves = 0
        while node.above is not None:
            node = node.above
            moves += 1
            if moves >= trunk.column_size:
                raise ValueError(f"Cycle ou trop de voisins au-dessus au pas {trunk.step}")
        return node

    def _column_nodes(self, trunk: TrunkNode) -> Iterator[Node]:
        """Valider la taille annoncée et la présence du tronc dans la chaîne."""
        if trunk.column_size < 1:
            raise ValueError(f"Taille de colonne invalide au pas {trunk.step}")
        node: Node | None = self._top(trunk)
        count = 0
        found_trunk = False
        # Descendre la colonne en comptant les nœuds rencontrés.
        while node is not None:
            count += 1
            if count > trunk.column_size or node.step != trunk.step:
                raise ValueError(f"Colonne incohérente au pas {trunk.step}")
            found_trunk = found_trunk or node is trunk
            yield node
            node = node.below
        if count != trunk.column_size or not found_trunk:
            raise ValueError(f"Colonne {trunk.step} : {count} nœuds ; "
                             f"taille annoncée {trunk.column_size}")

    def _is_pruned(self, node: Node) -> bool:
        """Lire le drapeau du moteur, ou déduire le pruning de la forme du nœud."""
        # Le repli permet d'inspecter d'anciens arbres sans drapeau.
        marker = getattr(node, "is_pruned", None)
        if marker is not None:
            return bool(marker)
        return (
            self.tree.transition_type(node.step) == "binomial"
            and node.next_mid is not None
            and node.next_up is None and node.next_down is None
        )

    # ------------------------------------------------------------------
    # Contrôles des liens : enfants, voisins et parents.
    # ------------------------------------------------------------------
    def _in_column(self, child: Node, trunk: TrunkNode, branch: str) -> bool:
        """Appeler dans l'ordre des parents, de haut en bas, pour chaque branche."""
        attribute = "_child_" + branch
        cursor = getattr(self, attribute)
        moves = 0
        # Le curseur ne remonte jamais : parcours linéaire de la colonne.
        while cursor is not None and cursor.spot_price > child.spot_price:
            cursor = cursor.below
            moves += 1
            if moves > trunk.column_size:
                raise ValueError("Cycle pendant la recherche d'un enfant")
        setattr(self, attribute, cursor)
        return cursor is child

    def _check_child(self, node: Node, trunk: TrunkNode, branch: str) -> None:
        """Un enfant existe si sa probabilité est non nulle, au pas suivant."""
        child = getattr(node, "next_" + branch)
        probability = getattr(node, "p_" + branch)
        if child is None:
            if abs(probability) > 1e-12:
                self._error(f"enfant {branch} manquant avec probabilité non nulle", node)
            return
        self.branch_count += 1
        if child.step != node.step + 1 or not self._in_column(child, trunk, branch):
            self._error(f"enfant {branch} hors de la colonne suivante", node)
        if not np.isfinite(child.spot_price) or child.spot_price <= 0:
            self._error(f"prix de l'enfant {branch} non positif ou non fini", node)

    def _check_recombination(self, node: Node) -> None:
        """Deux voisins binomiaux doivent partager un enfant commun."""
        # Un bord élagué conserve le même enfant partagé, mais ce lien
        # s'appelle next_mid et sa probabilité devient 1.
        lower = node.below
        upper_down = node.next_mid if self._is_pruned(node) else node.next_down
        lower_up = lower.next_mid if self._is_pruned(lower) else lower.next_up
        if upper_down is None or upper_down is not lower_up:
            self._error("recombinaison binomiale incorrecte", node)

    def _check_neighbors(self, node: Node) -> None:
        """Liens above/below réciproques et prix décroissants vers le bas."""
        if node.above is not None:
            if node.above.step != node.step or node.above.below is not node:
                self._error("lien above/below incorrect", node)
        if node.below is None:
            return
        if node.below.step != node.step or node.below.above is not node:
            self._error("lien below/above incorrect", node)
        if node.spot_price <= node.below.spot_price:
            self._error("prix mal ordonnés dans la colonne", node)
        if self.tree.transition_type(node.step) == "binomial":
            self._check_recombination(node)

    def _incoming(self, node: Node, previous: TrunkNode, branch: str) -> tuple[int, float]:
        """Nombre de branches entrantes et somme π(parent) × p(parent,enfant)."""
        attribute = "_parent_" + branch
        cursor = getattr(self, attribute)
        count, mass, moves = 0, 0.0, 0
        while cursor is not None:
            child = getattr(cursor, "next_" + branch)
            # Le parent courant doit être conservé pour l'enfant suivant :
            # son enfant est déjà situé plus bas que celui que l'on contrôle.
            if child is not None and child.spot_price < node.spot_price:
                break
            if child is node:
                count += 1
                mass += cursor.proba * getattr(cursor, "p_" + branch)
            cursor = cursor.below
            moves += 1
            if moves > previous.column_size:
                raise ValueError("Cycle pendant la recherche des parents")
        setattr(self, attribute, cursor)
        return count, mass

    def _check_root(self, node: Node) -> None:
        """À la date initiale, seule la racine existe, avec probabilité 1."""
        if node is not self.tree.root:
            self._error("nœud supplémentaire à la date initiale", node)
        if not self._close(node.proba, 1):
            self._error("la probabilité de la racine doit valoir 1", node)

    def _check_incoming_count(self, node: Node, previous: TrunkNode, count: int) -> None:
        """Nombre de parents attendu selon le type de transition."""
        kind = self.tree.transition_type(previous.step)
        # Avec pruning, un bord peut recevoir deux branches : règle non applicable.
        if kind == "binomial" and not self._previous_has_pruning:
            expected = 1 if node.above is None or node.below is None else 2
            if count != expected:
                self._error(f"{count} branches entrantes au lieu de {expected}", node)
        elif kind == "deterministic" and count != 1:
            self._error(f"{count} branches entrantes dans une transition déterministe", node)

    def _check_parents(self, node: Node, previous: TrunkNode | None) -> None:
        """La probabilité stockée doit égaler la somme des masses entrantes."""
        if not np.isfinite(node.proba) or not -1e-12 <= node.proba <= 1 + 1e-12:
            self._error("probabilité d'atteinte en dehors de [0,1]", node)
        if previous is None:
            self._check_root(node)
            return
        count, mass = 0, 0.0
        # Additionner les contributions des trois types de branches.
        for branch in BRANCHES:
            branch_count, branch_mass = self._incoming(node, previous, branch)
            count += branch_count
            mass += branch_mass
        if count == 0:
            self._error("nœud sans branche entrante", node)
        if not self._close(node.proba, mass):
            self._error(f"probabilité stockée {node.proba:.12g} ≠ somme entrante "
                        f"{mass:.12g}", node)
        self._check_incoming_count(node, previous, count)

    def _check_parameters(self) -> None:
        """Pas de temps, actualisation, racine et probabilités communes."""
        tree = self.tree
        expected_dt = ((tree.option.maturity - tree.model.pricing_date).days
                       / 365 / tree.model.nb_steps)
        if not self._close(tree.deltaT, expected_dt):
            self._error("pas de temps incorrect")
        if not self._close(tree.df, np.exp(-tree.market.rate * expected_dt)):
            self._error("facteur d'actualisation incorrect")
        if not self._close(tree.root.spot_price, tree.market.spot):
            self._error("prix de la racine différent du spot")
        # Les probabilités binomiales n'existent que si la volatilité est positive.
        if tree.market.vol > 0:
            up, down = tree.p_up, tree.p_down
            if not (0 <= up <= 1 and 0 <= down <= 1) or not self._close(up + down, 1):
                self._error("probabilités binomiales communes incorrectes")

    # ------------------------------------------------------------------
    # Contrôles des moments locaux (moyenne et variance d'un pas).
    # ------------------------------------------------------------------
    @staticmethod
    def _local_moments(node: Node) -> tuple[float, float]:
        """Var(S') par les écarts à la moyenne, plus stable que E[S'²]−E[S']²."""
        mean = 0.0
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                mean += getattr(node, "p_" + branch) * child.spot_price
        variance = 0.0
        # Second passage : variance autour de la moyenne obtenue.
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                variance += getattr(node, "p_" + branch) * (child.spot_price - mean) ** 2
        return mean, variance

    def _check_probability_sum(self, node: Node) -> None:
        """Chaque probabilité locale dans [0,1] et leur somme égale à 1."""
        probability_sum = 0.0
        for branch in BRANCHES:
            probability = getattr(node, "p_" + branch)
            probability_sum += probability
            if not np.isfinite(probability) or not -1e-12 <= probability <= 1 + 1e-12:
                self._error(f"probabilité {branch} en dehors de [0,1]", node)
        if not self._close(probability_sum, 1):
            self._error("la somme des probabilités locales ne vaut pas 1", node)

    def _check_pruned_node(self, node: Node, mean: float, variance: float) -> None:
        """Vérifier la forme d'une coupe puis mesurer son biais de moments."""
        self.pruned_node_count += 1
        if self.tree.transition_type(node.step) != "binomial" or (
                node.above is not None and node.below is not None):
            self._error("le pruning du moteur doit concerner un bord binomial", node)
        if node.proba >= self.tree.model.pruning_threshold:
            self._error("nœud élagué alors que sa probabilité n'est pas sous le seuil", node)
        if node.next_up is not None or node.next_down is not None or node.next_mid is None:
            self._error("pruning : un seul enfant central doit subsister", node)
        if not self._close(node.p_mid, 1) or not self._close(node.p_up + node.p_down, 0):
            self._error("pruning : la branche conservée doit avoir probabilité 1", node)
        # Biais volontaire : on le mesure au lieu de le signaler comme erreur.
        mean_bias = mean - node.forward()
        self.pruning_weighted_mean_bias += node.proba * mean_bias
        self.pruning_weighted_abs_mean_bias += node.proba * abs(mean_bias)
        self.pruning_weighted_variance_bias += node.proba * (variance - node.variance())
        self.pruning_max_abs_mean_bias = max(self.pruning_max_abs_mean_bias, abs(mean_bias))

    def _check_binomial_shape(self, node: Node) -> None:
        """Deux enfants au forward × / ÷ sqrt(alpha), probabilités communes."""
        tree = self.tree
        if node.next_mid is not None:
            self._error("branche centrale non marquée comme pruning "
                        "en transition binomiale", node)
        if node.next_up is None or node.next_down is None:
            self._error("deux enfants requis dans une transition binomiale", node)
        else:
            if not self._close(node.next_up.spot_price, node.forward() * tree.sqrtAlpha):
                self._error("prix de l'enfant haut binomial incorrect", node)
            if not self._close(node.next_down.spot_price, node.forward() / tree.sqrtAlpha):
                self._error("prix de l'enfant bas binomial incorrect", node)
        if not self._close(node.p_up, tree.p_up) or not self._close(node.p_down, tree.p_down):
            self._error("probabilités binomiales différentes des paramètres communs", node)

    def _check_trinomial_shape(self, node: Node) -> None:
        """Trois enfants voisins sur la grille, de rapport alpha."""
        up, mid, down = node.next_up, node.next_mid, node.next_down
        if up is None or mid is None or down is None:
            self._error("trois enfants requis au détachement discret", node)
            return
        if up.below is not mid or mid.below is not down:
            self._error("les trois enfants ne sont pas voisins sur la grille", node)
        if not self._close(up.spot_price / mid.spot_price, self.tree.alpha):
            self._error("rapport haut/milieu incorrect sur la grille", node)
        if not self._close(mid.spot_price / down.spot_price, self.tree.alpha):
            self._error("rapport milieu/bas incorrect sur la grille", node)

    def _check_deterministic_shape(self, node: Node) -> None:
        """Volatilité nulle : un seul enfant central, de probabilité 1."""
        if node.next_up is not None or node.next_down is not None or node.next_mid is None:
            self._error("transition déterministe : seul l'enfant central doit exister", node)
        if not self._close(node.p_mid, 1):
            self._error("transition déterministe : probabilité centrale différente de 1", node)

    def _check_moments(self, node: Node) -> None:
        """Moyenne = forward et variance = variance du modèle (hors pruning)."""
        self._check_probability_sum(node)
        mean, variance = self._local_moments(node)
        if self._is_pruned(node):
            # Ces deux moments sont volontairement altérés par l'élagage.
            self._check_pruned_node(node, mean, variance)
            return
        if not self._close(mean, node.forward()):
            self._error("espérance différente du forward après dividendes", node)
        if not self._close(variance, node.variance()):
            self._error("variance différente de la variance prévue par le modèle", node)
        # Contrôle de forme propre à chaque type de transition.
        kind = self.tree.transition_type(node.step)
        if kind == "binomial":
            self._check_binomial_shape(node)
        elif kind == "trinomial":
            self._check_trinomial_shape(node)
        elif kind == "deterministic":
            self._check_deterministic_shape(node)

    # ------------------------------------------------------------------
    # Contrôles de la valorisation (après tree.price()).
    # ------------------------------------------------------------------
    def _check_terminal_value(self, node: Node, intrinsic: float) -> None:
        """À maturité : valeur = payoff, exercice si payoff positif."""
        if not self._close(node.option_value, intrinsic):
            self._error("valeur terminale différente du payoff", node)
        if hasattr(node, "exercise_now") and bool(node.exercise_now) != (intrinsic > 0):
            self._error("décision terminale incompatible avec un payoff positif", node)

    def _recomputed_continuation(self, node: Node) -> float | None:
        """Recalculer la continuation ; None si un enfant n'est pas valorisé."""
        continuation = 0.0
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is None:
                continue
            if child.option_value is None:
                self._error("enfant non valorisé alors que le parent l'est", node)
                return None
            continuation += getattr(node, "p_" + branch) * child.option_value
        return continuation * self.tree.df

    def _check_valuation(self, node: Node) -> None:
        """Contrôler une valorisation déjà calculée ; ne pas imposer price()."""
        if node.option_value is None:
            return
        if not np.isfinite(node.option_value):
            self._error("valeur de l'option non finie", node)
            return
        intrinsic = self.tree.option.intrinsic_value(node.spot_price)
        if node.step == self.tree.model.nb_steps:
            self._check_terminal_value(node, intrinsic)
            return
        continuation = self._recomputed_continuation(node)
        if continuation is not None:
            self._check_decision(node, intrinsic, continuation)

    def _check_decision(self, node: Node, intrinsic: float, continuation: float) -> None:
        """Refaire la décision du moteur avec la même tolérance et comparer."""
        american = self.tree.option.exercise_type.lower() == "american"
        tolerance = self.tree.exercise_tolerance(intrinsic, continuation)
        should_exercise = american and intrinsic > 0 and intrinsic > continuation + tolerance
        if not self._close(node.option_value, intrinsic if should_exercise else continuation):
            self._error("valeur incompatible avec continuation/exercice", node)
        stored = getattr(node, "continuation_value", None)
        if stored is not None and not self._close(stored, continuation):
            self._error("valeur de continuation stockée incorrecte", node)
        decision = getattr(node, "exercise_now", None)
        if decision is not None and bool(decision) != should_exercise:
            self._error("décision d'exercice incorrecte (à égalité numérique, conserver)", node)

    # ------------------------------------------------------------------
    # Contrôle complet d'une colonne puis de l'arbre.
    # ------------------------------------------------------------------
    def _start_cursors(self, trunk: TrunkNode) -> None:
        """Placer les curseurs en haut des colonnes suivante et précédente."""
        if trunk.next_trunk is not None:
            first = self._top(trunk.next_trunk)
            self._child_up = self._child_mid = self._child_down = first
        if trunk.previous_trunk is not None:
            first = self._top(trunk.previous_trunk)
            self._parent_up = self._parent_mid = self._parent_down = first

    def _check_transitions(self, node: Node, trunk: TrunkNode) -> None:
        """Avant maturité : enfants et moments ; à maturité : aucune branche."""
        if node.step < self.tree.model.nb_steps:
            if not isinstance(trunk.next_trunk, TrunkNode):
                raise ValueError("Colonne suivante absente")
            for branch in BRANCHES:
                self._check_child(node, trunk.next_trunk, branch)
            self._check_moments(node)
            return
        # Dernière colonne : on accumule E[S(T)] pour le contrôle K=0.
        self.terminal_spot_expectation += node.proba * node.spot_price
        if any(getattr(node, "next_" + branch) is not None for branch in BRANCHES):
            self._error("branche après maturité", node)
        if self._is_pruned(node):
            self._error("un nœud terminal n'a pas de transition à élaguer", node)

    def _check_node(self, node: Node, trunk: TrunkNode) -> None:
        """Tous les contrôles d'un nœud, dans l'ordre de lecture."""
        if node.tree is not self.tree:
            self._error("référence vers un autre arbre", node)
        if not np.isfinite(node.spot_price) or node.spot_price <= 0:
            self._error("prix du nœud non positif ou non fini", node)
        self._check_neighbors(node)
        self._check_parents(node, trunk.previous_trunk)
        self._check_transitions(node, trunk)
        self._check_valuation(node)

    def _check_column(self, trunk: TrunkNode) -> None:
        """Contrôler chaque nœud et la conservation de masse de la colonne."""
        self._start_cursors(trunk)
        mass = 0.0
        current_has_pruning = False
        for node in self._column_nodes(trunk):
            self.node_count += 1
            mass += node.proba
            current_has_pruning = current_has_pruning or self._is_pruned(node)
            self._check_node(node, trunk)
        # La somme des masses d'une colonne doit toujours valoir 1.
        if not self._close(mass, 1):
            self._error(f"colonne {trunk.step} : somme des probabilités = {mass:.12g}, "
                        "au lieu de 1")
        self._previous_has_pruning = current_has_pruning

    def _check_zero_strike(self) -> None:
        """Identité call européen K=0 = forward actualisé, seulement sans pruning.

        Le changement d'option est temporaire. Un finally remet l'option de
        l'utilisateur et recalcule ses valeurs/décisions dans tous les nœuds.
        """
        original = self.tree.option
        try:
            self.tree.option = Option(0, original.maturity, "Call", "European")
            price = self.tree.price()
            expected = self.tree.expected_zero_strike()
            if not self._close(price, expected):
                self._error("le call européen de strike nul diffère du forward actualisé")
            print(f"Call européen K=0 : {price:.10f} ; forward actualisé : {expected:.10f}")
        finally:
            self.tree.option = original
            self.tree.price()

    def _report_pruning(self) -> None:
        """Avec pruning : afficher les biais mesurés au lieu du test K=0."""
        tree = self.tree
        discounted_mean = tree.df**tree.model.nb_steps * self.terminal_spot_expectation
        self.discounted_terminal_mean_bias = discounted_mean - tree.expected_zero_strike()
        print(f"PRUNING : {self.pruned_node_count} transitions monomiales ; "
              f"Σ π·|Δm| = {self.pruning_weighted_abs_mean_bias:.6g} ; "
              f"max |Δm| = {self.pruning_max_abs_mean_bias:.6g}.")
        print(f"Biais de E[S(T)] actualisé : {self.discounted_terminal_mean_bias:+.10g} ; "
              f"Σ π·Δvariance = {self.pruning_weighted_variance_bias:+.6g}.")
        print("Information : les moments des nœuds coupés sont altérés ; "
              "test exact K=0 non appliqué.")

    def check(self) -> int:
        """Afficher le diagnostic et renvoyer le nombre d'erreurs (remis à zéro)."""
        self._reset_counters()
        try:
            self._check_parameters()
            for trunk in self._columns():
                self._check_column(trunk)
            # Contrôle global final, seulement si la structure est saine.
            if self.errors == 0:
                if self.pruned_node_count == 0:
                    self._check_zero_strike()
                else:
                    self._report_pruning()
        except (ValueError, AttributeError, TypeError, ZeroDivisionError, OverflowError) as exc:
            self._error(str(exc))
        print(f"DEBUG : {self.node_count} nœuds parcourus, {self.branch_count} branches")
        print(f"Bilan : {self.errors} erreur(s) détectée(s)")
        return self.errors

    # ------------------------------------------------------------------
    # Dessin SVG : préparation (couleurs, échelle) puis écriture.
    # ------------------------------------------------------------------
    def _value(self, node: Node, color_by: str) -> float:
        """Grandeur utilisée pour colorer le nœud."""
        if color_by == "spot":
            return node.spot_price
        if color_by == "probability":
            return node.proba
        if node.option_value is None:
            raise ValueError("Appeler tree.price() avant de colorer les valeurs de l'option.")
        return node.option_value

    @staticmethod
    def validate_color(color_by: str) -> None:
        """Refuser un critère de couleur inconnu."""
        if color_by not in ("spot", "option", "probability"):
            raise ValueError("color_by doit être 'spot', 'option' ou 'probability'.")

    def _color_scale(self, color_by: str) -> Normalize:
        """Échelle de couleur entre la plus petite et la plus grande valeur."""
        minimum, maximum = math.inf, -math.inf
        for trunk in self._columns():
            for node in self._column_nodes(trunk):
                value = self._value(node, color_by)
                minimum, maximum = min(minimum, value), max(maximum, value)
        return Normalize(minimum, maximum if maximum > minimum else minimum + 1)

    def _decision(self, node: Node) -> str:
        """Libellé de la décision du moteur affiché dans la boîte du nœud."""
        if node.step == self.tree.model.nb_steps:
            payoff = self.tree.option.intrinsic_value(node.spot_price)
            return "MATURITÉ : EXERCER" if payoff > 0 else "EXPIRATION SANS VALEUR"
        action = "EXERCER" if getattr(node, "exercise_now", False) else "CONSERVER"
        return action + (" · PRUNING" if self._is_pruned(node) else "")

    def _prepare_drawing(self, color_by: str, max_nodes: int, run_checks: bool) -> int:
        """Valider les options du dessin, compter les nœuds et pricer si besoin."""
        self.validate_color(color_by)
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes doit être un entier strictement positif.")
        # Compter par les liens validés, sans stocker les nœuds dans une liste.
        count = sum(1 for trunk in self._columns() for _ in self._column_nodes(trunk))
        if count > max_nodes:
            raise ValueError(
                f"Dessin refusé : {count} nœuds, limite explicite {max_nodes}. "
                "Réduire l'arbre ou augmenter max_nodes ; check() n'a pas cette limite."
            )
        if self.tree.root.option_value is None:
            self.tree.price()
        if run_checks and self.check() != 0:
            raise ValueError("Dessin interrompu : corriger les erreurs signalées puis relancer.")
        return count

    def _log_range(self) -> tuple[float, float, float]:
        """Plus petit/grand log(S) et plus petit écart entre deux voisins."""
        min_log, max_log, min_gap = math.inf, -math.inf, math.inf
        for trunk in self._columns():
            for node in self._column_nodes(trunk):
                log_spot = math.log(node.spot_price)
                min_log, max_log = min(min_log, log_spot), max(max_log, log_spot)
                if node.below is not None:
                    min_gap = min(min_gap, log_spot - math.log(node.below.spot_price))
        if min_gap <= 0:
            raise ValueError("Le dessin nécessite des spots strictement ordonnés et positifs.")
        return min_log, max_log, min_gap

    def _canvas_size(self) -> tuple[float, float, float, float]:
        """Échelle verticale, largeur, hauteur et log maximal du dessin."""
        min_log, max_log, min_gap = self._log_range()
        # Au moins 92 pixels entre les centres de deux voisins de colonne.
        # En volatilité nulle il n'y a qu'un nœud par date : échelle de secours.
        if math.isfinite(min_gap):
            scale_y = 92 / min_gap
        else:
            scale_y = 300 / max(max_log - min_log, 1e-6)
        width = 2 * _MARGIN_X + self.tree.model.nb_steps * _STEP_WIDTH
        height = max(580.0, _TOP_Y + (max_log - min_log) * scale_y + 180)
        if max(width, height) > 100_000:
            raise ValueError("Le canevas dépasserait 100 000 pixels : "
                             "réduire l'arbre pour le dessin.")
        return scale_y, width, height, max_log

    def _write_header(self, canvas: _SvgCanvas, width: float, height: float,
                      count: int) -> None:
        """En-tête SVG, styles, titre et légende du dessin."""
        tree = self.tree
        canvas.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        canvas.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
                      f'height="{height:.0f}" viewBox="0 0 {width:.2f} {height:.2f}" '
                      'role="img" aria-labelledby="title description">\n')
        canvas.write('<title id="title">Arbre de valorisation : construction, '
                      'exercice et pruning</title>\n')
        canvas.write(_SVG_DESCRIPTION + _SVG_DEFS + _SVG_STYLE)
        canvas.write(f'<rect width="{width:.2f}" height="{height:.2f}" fill="#f8fafc"/>\n')
        # Titre puis lignes d'aide à gauche du dessin.
        canvas.text(36, 38, f"{tree.option.type} {tree.option.exercise_type} · "
                    f"K={tree.option.strike:g} · {tree.model.nb_steps} pas · "
                    f"{count} nœuds", "heading", "start")
        for py, label, css in _LEGEND_LINES:
            canvas.text(36, py, label, css, "start")
        canvas.text(36, 133, f"r={tree.market.rate:.2%} · σ={tree.market.vol:.2%} · "
                    f"q={tree.market.dividend_yield:.2%} · dividende cash="
                    f"{tree.market.dividend_amount:g} € · seuil pruning="
                    f"{tree.model.pruning_threshold:g}", "small", "start")

    def _write_columns(self, canvas: _SvgCanvas, height: float) -> None:
        """Lignes verticales des dates, taille et masse de chaque colonne."""
        for trunk in self._columns():
            px = canvas.x(trunk)
            canvas.write(f'<line x1="{px:.2f}" y1="192" x2="{px:.2f}" '
                          f'y2="{height - 90:.2f}" stroke="#e2e8f0"/>\n')
            canvas.text(px, 164, f"t{trunk.step} = {trunk.step * self.tree.deltaT:.3f} an")
            column_mass = sum(node.proba for node in self._column_nodes(trunk))
            canvas.text(px, 183, f"{trunk.column_size} nœuds · Σπ={column_mass:.6f}", "small")
            # Repère rouge à la fin du pas qui contient le dividende cash.
            if trunk.step == self.tree.dividend_step:
                canvas.write(f'<line x1="{px:.2f}" y1="192" x2="{px:.2f}" '
                              f'y2="{height - 90:.2f}" stroke="#dc2626" '
                              'stroke-dasharray="5 5" opacity="0.45"/>\n')
                canvas.text(px, 205, "FIN DU PAS DIVIDENDE CASH", "decision")

    def _write_forward(self, canvas: _SvgCanvas) -> None:
        """Courbe du forward théorique (repère, sans biais de pruning)."""
        previous_x = previous_y = None
        for trunk in self._columns():
            theoretical = self.tree.mean_at_time(trunk.step * self.tree.deltaT)
            if theoretical <= 0:
                continue
            px, py = canvas.x(trunk), canvas.y_spot(theoretical)
            # Un segment entre deux dates consécutives.
            if previous_x is not None:
                canvas.write(f'<path class="forward" d="M {previous_x:.2f} '
                              f'{previous_y:.2f} L {px:.2f} {py:.2f}"/>\n')
            previous_x, previous_y = px, py

    def _write_edge(self, canvas: _SvgCanvas, node: Node, branch: str) -> None:
        """Une flèche parent → enfant avec sa probabilité conditionnelle."""
        child = getattr(node, "next_" + branch)
        if child is None:
            return
        probability = getattr(node, "p_" + branch)
        pruned = self._is_pruned(node)
        # Orange épais et pointillé pour une branche issue du pruning.
        stroke = "#ea580c" if pruned else "#94a3b8"
        thickness = 2.8 if pruned else 1.1
        dash = ' stroke-dasharray="5 3"' if pruned or probability == 0 else ""
        x1, x2 = canvas.x(node) + _BOX_WIDTH / 2, canvas.x(child) - _BOX_WIDTH / 2
        y1, y2 = canvas.y(node), canvas.y(child)
        canvas.write(f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="{thickness}" '
                      f'marker-end="url(#arrow)"{dash}/>\n')
        label = "p=1" if self._close(probability, 1) else f"p={probability:.3g}"
        canvas.text((x1 + x2) / 2, (y1 + y2) / 2 - 5, label, "edge-label")

    def _write_edges(self, canvas: _SvgCanvas, show_neighbors: bool) -> None:
        """Toutes les flèches AVANT les boîtes, qui masquent ainsi les traits."""
        for trunk in self._columns():
            for node in self._column_nodes(trunk):
                for branch in BRANCHES:
                    self._write_edge(canvas, node, branch)
                # Liens verticaux facultatifs entre voisins d'une colonne.
                if show_neighbors and node.below is not None:
                    canvas.write(f'<path d="M {canvas.x(node):.2f} {canvas.y(node):.2f} '
                                  f'L {canvas.x(node):.2f} {canvas.y(node.below):.2f}" '
                                  'stroke="#64748b" stroke-dasharray="2 4" opacity="0.45"/>\n')

    def _node_tooltip(self, node: Node) -> str:
        """Texte affiché au survol : valeurs, décision et moments locaux."""
        detail = (f"Pas {node.step} ; S={node.spot_price:.12g} ; V={node.option_value:.12g} ; "
                  f"π={node.proba:.12g} ; {self._decision(node)}")
        detail += f" ; payoff={self.tree.option.intrinsic_value(node.spot_price):.12g}"
        continuation = getattr(node, "continuation_value", None)
        if continuation is not None:
            detail += f" ; continuation={continuation:.12g}"
        else:
            detail += " ; continuation=sans objet à maturité"
        # Moments du pas suivant et leur écart au modèle (biais de pruning).
        if node.step < self.tree.model.nb_steps:
            mean, variance = self._local_moments(node)
            detail += (f" ; moyenne={mean:.12g} ; Δm={mean-node.forward():+.6g} ; "
                       f"variance={variance:.12g} ; Δvar={variance-node.variance():+.6g}")
        return detail

    def _write_box(self, canvas: _SvgCanvas, node: Node, fill: str, text_color: str) -> None:
        """Rectangle coloré du nœud (double contour pour le tronc)."""
        px, py = canvas.x(node), canvas.y(node)
        exercising = (node.step < self.tree.model.nb_steps
                      and getattr(node, "exercise_now", False))
        # Contour : rouge = exercice anticipé, orange = pruning, gris sinon.
        border = "#dc2626" if exercising else "#ea580c" if self._is_pruned(node) else "#475569"
        stroke_width = 3 if exercising or self._is_pruned(node) else 1.2
        canvas.write(f'<rect x="{px-_BOX_WIDTH/2:.2f}" y="{py-_BOX_HEIGHT/2:.2f}" '
                      f'width="{_BOX_WIDTH}" height="{_BOX_HEIGHT}" rx="9" fill="{fill}" '
                      f'stroke="{border}" stroke-width="{stroke_width}"/>\n')
        if isinstance(node, TrunkNode):
            canvas.write(f'<rect x="{px-_BOX_WIDTH/2+4:.2f}" y="{py-_BOX_HEIGHT/2+4:.2f}" '
                          f'width="{_BOX_WIDTH-8}" height="{_BOX_HEIGHT-8}" rx="6" '
                          f'fill="none" stroke="{text_color}" stroke-opacity="0.65"/>\n')

    def _write_node(self, canvas: _SvgCanvas, node: Node, norm: Normalize,
                    color_by: str) -> None:
        """Boîte du nœud : couleur, contour, infobulle et quatre lignes de texte."""
        value = self._value(node, color_by)
        fill = to_hex(plt.get_cmap("viridis")(norm(value)))
        text_color = "#ffffff" if norm(value) < 0.6 else "#111827"
        canvas.write(f'<g id="node-{node.step}-{id(node)}">\n')
        canvas.write(f'<title>{escape(self._node_tooltip(node))}</title>\n')
        self._write_box(canvas, node, fill, text_color)
        canvas.write(f'<g style="fill:{text_color}">\n')
        # Un style explicite sur chaque texte prévaut sur la règle
        # générale CSS « text{fill:...} » du document.
        px, py = canvas.x(node), canvas.y(node)
        for offset, label, size in (
            (-18, f"S = {node.spot_price:.4f}", 12),
            (-2, f"V = {node.option_value:.5f}", 12),
            (14, f"π = {node.proba:.5g}", 11),
            (29, self._decision(node), 9),
        ):
            canvas.write(f'<text x="{px:.2f}" y="{py+offset:.2f}" text-anchor="middle" '
                          f'style="fill:{text_color};font-size:{size}px">'
                          f'{escape(label)}</text>\n')
        canvas.write('</g></g>\n')

    def _write_svg(self, stream: TextIO, color_by: str, show_neighbors: bool,
                   count: int) -> None:
        """Écrire tout le document SVG dans un flux texte ouvert."""
        norm = self._color_scale(color_by)
        scale_y, width, height, max_log = self._canvas_size()
        canvas = _SvgCanvas(stream, max_log, scale_y)
        # Ordre d'écriture = ordre d'empilement : fond, flèches, puis boîtes.
        self._write_header(canvas, width, height, count)
        self._write_columns(canvas, height)
        self._write_forward(canvas)
        self._write_edges(canvas, show_neighbors)
        for trunk in self._columns():
            for node in self._column_nodes(trunk):
                self._write_node(canvas, node, norm, color_by)
        color_name = {"spot": "S", "option": "V", "probability": "π"}[color_by]
        canvas.text(36, height - 50, f"Couleur : {color_name}, minimum {norm.vmin:.6g} → "
                    f"maximum {norm.vmax:.6g}. Zoomer dans le navigateur pour lire les "
                    "détails ; survoler un nœud pour ses moments.", "small", "start")
        canvas.write('</svg>\n')

    def export_svg(self, path: str | Path, color_by: str = "option",
                   show_neighbors: bool = False, max_nodes: int = DEFAULT_MAX_DRAW_NODES,
                   run_checks: bool = True) -> Path:
        """Écrire un SVG autonome, vectoriel et zoomable, et renvoyer son chemin.

        L'axe vertical est logarithmique : deux voisins de rapport alpha ont
        le même espacement. Chaque boîte contient S, V et π (probabilité
        risque neutre d'atteindre ce nœud). Les probabilités p des flèches
        sont conditionnelles au nœud de départ. Le pruning apparaît en
        orange, l'exercice anticipé en rouge, et le tronc avec un double
        contour. Au survol, un titre donne les moments locaux.
        """
        count = self._prepare_drawing(color_by, max_nodes, run_checks)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Le fichier est écrit au fil de l'eau, sans garder les nœuds en mémoire.
        with destination.open("w", encoding="utf-8") as svg:
            self._write_svg(svg, color_by, show_neighbors, count)
        print("SVG enregistré :", destination)
        return destination


def main() -> None:
    """Exemple reproductible : contrôle puis export SVG d'un petit arbre."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="Nombre de pas (défaut : 20).")
    parser.add_argument("--pruning-threshold", type=float, default=1e-3,
                        help="Seuil volontairement visible ; 0 désactive le pruning.")
    parser.add_argument("--output", type=Path, default=Path("debug_arbre.svg"),
                        help="Fichier SVG de destination.")
    parser.add_argument("--color-by", choices=("spot", "option", "probability"),
                        default="option")
    parser.add_argument("--show-neighbors", action="store_true",
                        help="Dessiner aussi les liens verticaux.")
    args = parser.parse_args()
    # Put américain avec dividende cash : exercice et pas trinomial visibles.
    model = Model(pricing_date=dt.date(2026, 9, 15), nb_steps=args.steps,
                  pruning_threshold=args.pruning_threshold)
    market = Market(spot=100, rate=0.03, vol=0.20, dividend_yield=0.02,
                    dividend_amount=3.0, ex_dividend_date=dt.date(2027, 3, 15))
    option = Option(strike=101, maturity=dt.date(2027, 9, 2), type="Put",
                    exercise_type="American")
    tree = Tree(model=model, option=option, market=market).build()
    tree.price()
    DebugTree(tree).export_svg(args.output, args.color_by, args.show_neighbors)


if __name__ == "__main__":
    main()
