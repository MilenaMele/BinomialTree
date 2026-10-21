# =============================================================================
# Dessin SVG compact de l'arbre, pour un grand nombre de pas (100 et plus).
#
# À quoi sert ce module ?
#     Le dessin détaillé de debug_arbre.py écrit S, V et π dans une boîte par
#     nœud : il devient illisible au-delà d'une quarantaine de pas. Ici chaque
#     nœud est un simple point, comme sur la figure du cours (p. 22) :
#
#     * position : pas de temps en abscisse, log(S) en ordonnée ;
#     * couleur  : valeur de l'option, spot ou probabilité (échelle viridis) ;
#     * taille   : probabilité d'atteindre le nœud (gros au centre) ;
#     * contour rouge : exercice anticipé ; branche orange : pruning.
#
#     Les détails (S, V, π, décision) apparaissent au survol d'un point.
#     Les branches sont regroupées en quelques chemins SVG pour garder un
#     fichier léger, même avec plusieurs milliers de nœuds.
# =============================================================================
"""Dessin SVG compact (un point par nœud) pour les grands arbres."""

# Imports : bibliothèque standard, couleurs Matplotlib puis moteur.
from __future__ import annotations

import math
from typing import Iterator
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex

from arbre_binomial import BRANCHES, Node, Tree

# Taille du canevas et marges, en pixels.
_WIDTH, _HEIGHT = 1600.0, 900.0
_LEFT, _RIGHT, _TOP, _BOTTOM = 70.0, 40.0, 110.0, 70.0
# Rayon des points : du plus petit (bords) au plus grand (racine).
_MIN_RADIUS, _MAX_RADIUS = 0.8, 9.0
# Au-delà, le fichier serait trop lourd pour un navigateur.
MAX_COMPACT_NODES = 50_000


# Dessin compact d'un arbre déjà construit et pricé
class CompactTreeSvg:
    """Produire le SVG compact d'un arbre en suivant uniquement ses liens."""

    def __init__(self, tree: Tree, color_by: str = "option") -> None:
        # L'arbre doit être pricé pour colorer par valeur et marquer l'exercice.
        if tree.root.option_value is None:
            tree.price()
        self.tree = tree
        self.color_by = color_by
        self._min_log, self._max_log = self._log_range()
        self.norm = self._color_scale()

    # Générateur des nœuds, colonne par colonne (aucune liste stockée)
    def _nodes(self) -> Iterator[Node]:
        """Générateur des nœuds, colonne par colonne (aucune liste stockée)."""
        return (node for trunk in self.tree.iter_columns()
                for node in self.tree.iter_column(trunk))

    # Plus petit et plus grand log(S) de l'arbre
    def _log_range(self) -> tuple[float, float]:
        """Plus petit et plus grand log(S) de l'arbre."""
        logs = [math.log(node.spot_price) for node in self._nodes()]
        if len(logs) > MAX_COMPACT_NODES:
            raise ValueError(f"Dessin refusé : {len(logs)} nœuds (limite {MAX_COMPACT_NODES}).")
        low, high = min(logs), max(logs)
        return low, high if high > low else low + 1.0

    # Grandeur utilisée pour la couleur du point
    def _value(self, node: Node) -> float:
        """Grandeur utilisée pour la couleur du point."""
        if self.color_by == "spot":
            return node.spot_price
        if self.color_by == "probability":
            return node.proba
        return node.option_value

    # Échelle de couleur entre la plus petite et la plus grande valeur
    def _color_scale(self) -> Normalize:
        """Échelle de couleur entre la plus petite et la plus grande valeur."""
        values = [self._value(node) for node in self._nodes()]
        low, high = min(values), max(values)
        return Normalize(low, high if high > low else low + 1.0)

    # Coordonnées : temps en abscisse, log(S) en ordonnée
    def _x(self, node: Node) -> float:
        """Abscisse du nœud selon son pas."""
        usable = _WIDTH - _LEFT - _RIGHT
        return _LEFT + node.step * usable / self.tree.model.nb_steps

    # Ordonnée logarithmique du prix
    def _y(self, spot: float) -> float:
        """Ordonnée logarithmique du prix."""
        usable = _HEIGHT - _TOP - _BOTTOM
        share = (self._max_log - math.log(spot)) / (self._max_log - self._min_log)
        return _TOP + share * usable

    # Toutes les branches d'un type dans un seul attribut d (fichier léger)
    def _edges_path(self, pruned: bool) -> str:
        """Toutes les branches d'un type dans un seul attribut d (fichier léger)."""
        parts = []
        for node in self._nodes():
            if bool(node.is_pruned) != pruned:
                continue
            for branch in BRANCHES:
                child = getattr(node, "next_" + branch)
                if child is not None:
                    parts.append(f"M{self._x(node):.1f} {self._y(node.spot_price):.1f}"
                                 f"L{self._x(child):.1f} {self._y(child.spot_price):.1f}")
        return "".join(parts)

    # Rayon proportionnel à la racine carrée de la probabilité du nœud
    def _radius(self, node: Node) -> float:
        """Rayon proportionnel à la racine carrée de la probabilité du nœud."""
        return _MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * math.sqrt(max(node.proba, 0.0))

    # Exercice anticipé (avant maturité) décidé par le moteur
    def _is_exercised(self, node: Node) -> bool:
        """Exercice anticipé (avant maturité) décidé par le moteur."""
        return node.step < self.tree.model.nb_steps and bool(node.exercise_now)

    # Un point coloré, avec ses détails dans une infobulle
    def _circle(self, node: Node) -> str:
        """Un point coloré, avec ses détails dans une infobulle."""
        fill = to_hex(plt.get_cmap("viridis")(self.norm(self._value(node))))
        stroke = ' stroke="#dc2626" stroke-width="1.2"' if self._is_exercised(node) else ""
        detail = (f"Pas {node.step} · S={node.spot_price:.4f} · V={node.option_value:.4f} · "
                  f"π={node.proba:.4g}" + (" · EXERCER" if self._is_exercised(node) else ""))
        return (f'<circle cx="{self._x(node):.1f}" cy="{self._y(node.spot_price):.1f}" '
                f'r="{self._radius(node):.2f}" fill="{fill}"{stroke}>'
                f'<title>{escape(detail)}</title></circle>\n')

    # Éléments fixes : titre, légende, repère du dividende
    def _header(self) -> str:
        """Titre, ligne de paramètres et légende des symboles."""
        tree = self.tree
        title = (f"{tree.option.type} {tree.option.exercise_type} · K={tree.option.strike:g} · "
                 f"{tree.model.nb_steps} pas · {tree.count_nodes()} nœuds")
        params = (f"S0={tree.market.spot:g} · r={tree.market.rate:.2%} · σ={tree.market.vol:.2%}"
                  f" · q={tree.market.dividend_yield:.2%} · dividende cash="
                  f"{tree.market.dividend_amount:g} € · seuil pruning="
                  f"{tree.model.pruning_threshold:g}")
        legend = ("Point = nœud · taille = probabilité π · couleur = "
                  f"{self._color_label()} · contour rouge = exercice anticipé · "
                  "branche orange = pruning · survoler un point pour ses valeurs")
        return (f'<text x="{_LEFT}" y="34" font-size="22" font-weight="700">'
                f'{escape(title)}</text>\n'
                f'<text x="{_LEFT}" y="60" font-size="13">{escape(params)}</text>\n'
                f'<text x="{_LEFT}" y="82" font-size="12" fill="#475569">'
                f'{escape(legend)}</text>\n')

    # Nom de la grandeur colorée, avec son minimum et son maximum
    def _color_label(self) -> str:
        """Nom de la grandeur colorée, avec son minimum et son maximum."""
        name = {"spot": "spot S", "probability": "probabilité π"}.get(self.color_by,
                                                                       "valeur V")
        return f"{name} (violet {self.norm.vmin:.4g} → jaune {self.norm.vmax:.4g})"

    # Ligne pointillée à la fin du pas qui contient le dividende cash
    def _dividend_line(self) -> str:
        """Ligne pointillée à la fin du pas qui contient le dividende cash."""
        step = self.tree.dividend_step
        if step is None:
            return ""
        x = _LEFT + step * (_WIDTH - _LEFT - _RIGHT) / self.tree.model.nb_steps
        return (f'<line x1="{x:.1f}" y1="{_TOP - 10}" x2="{x:.1f}" y2="{_HEIGHT - _BOTTOM}" '
                'stroke="#dc2626" stroke-dasharray="6 5" opacity="0.6"/>\n'
                f'<text x="{x + 6:.1f}" y="{_TOP - 14}" font-size="12" fill="#dc2626">'
                'dividende cash</text>\n')

    # Axe du temps : pas 0, milieu et maturité
    def _axis(self) -> str:
        """Axe du temps : pas 0, milieu et maturité."""
        n = self.tree.model.nb_steps
        y = _HEIGHT - _BOTTOM + 30
        labels = ""
        # Trois repères suffisent : date de calcul, milieu, maturité.
        for step, anchor in ((0, "start"), (n // 2, "middle"), (n, "end")):
            x = _LEFT + step * (_WIDTH - _LEFT - _RIGHT) / n
            labels += (f'<text x="{x:.1f}" y="{y}" font-size="12" text-anchor="{anchor}">'
                       f'pas {step} · t={step * self.tree.deltaT:.2f} an</text>\n')
        return labels

    # Document SVG complet, sous forme de texte
    def to_svg(self) -> str:
        """Document SVG complet, sous forme de texte."""
        body = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH:.0f}" '
            f'height="{_HEIGHT:.0f}" viewBox="0 0 {_WIDTH:.0f} {_HEIGHT:.0f}" '
            'font-family="DejaVu Sans,Arial,sans-serif">\n',
            f'<rect width="{_WIDTH:.0f}" height="{_HEIGHT:.0f}" fill="#f8fafc"/>\n',
            self._header(), self._dividend_line(), self._axis(),
            # Branches d'abord, en dessous des points.
            f'<path d="{self._edges_path(False)}" stroke="#94a3b8" stroke-width="0.35" '
            'fill="none" opacity="0.7"/>\n',
            f'<path d="{self._edges_path(True)}" stroke="#ea580c" stroke-width="1.2" '
            'fill="none"/>\n',
        ]
        body.extend(self._circle(node) for node in self._nodes())
        body.append("</svg>\n")
        return "".join(body)
