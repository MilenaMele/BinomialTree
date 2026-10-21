# =============================================================================
# Interface Streamlit du pricer.
#
# Il affiche l'application web : formulaire des paramètres dans la barre
# latérale, puis huit pages (prix, valeur & exercice, grecques, arbre, convergence,
# pruning, dividendes & taux, guide). 
# Appelle ``interface_calculs`` (qui appelle le moteur ``arbre_binomial``)
# et dessine les tableaux renvoyés avec Plotly.
#
# Lancer avec : python -m streamlit run app.py
# =============================================================================
"""Interface Streamlit du pricer (seul fichier qui utilise Streamlit)."""

# Imports : bibliothèque standard, paquets externes puis modules du projet.
from __future__ import annotations

import datetime as dt
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from functools import partial
from typing import Callable

# Un serveur web ne doit pas ouvrir de fenêtre Matplotlib, même sous Windows.
# Définir le backend AVANT d'importer les modules de calcul qui utilisent pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from interface_calculs import (
    EXERCISE_SCENARIOS, Parameters, calculate_summary, check_tree, convergence_study,
    debug_tree, dividend_study, exercise_comparison, greeks_by_spot, greeks_summary,
    has_active_cash, pruning_study, rate_study, strike_study, taylor_check,
)
from interface_exports import profile_archive, scenario_json

# Pages de la navigation, palette des graphiques et bornes des dates.
PAGES = (
    "Prix & comparaison", "Valeur & exercice", "Grecques & couverture",
    "Arbre & diagnostics", "Convergence", "Pruning", "Dividendes & taux", "Guide",
)
COLORS = ("#2563eb", "#0891b2", "#f59e0b", "#8b5cf6", "#059669")
GITHUB = "https://github.com/BChalet/benjaminchalet"
DATE_MIN, DATE_MAX = dt.date(1900, 1, 1), dt.date(2100, 12, 31)
PRESETS = ("Put américain / dividende cash", "Call européen / Black-Scholes",
           "Call américain / taux négatif")
EXERCISE_LABELS = {"European": "européen", "American": "américain"}
# Les trois types de contrats sont cochés par défaut.
DEFAULT_SCENARIOS = ("american_dividends", "american_no_dividend", "european_no_dividend")
CALCULATION_ERRORS = (ValueError, ArithmeticError, RuntimeError, OSError)


# =============================================================================
# 1. FORMULAIRE DES PARAMÈTRES (barre latérale)
# =============================================================================
def _preset(name: str) -> Parameters:
    """Trois exemples valides, modifiables ensuite champ par champ."""
    base = Parameters(
        pricing_date=dt.date(2026, 9, 15), maturity=dt.date(2027, 9, 2),
        spot=100.0, strike=101.0, rate=0.07, vol=0.30, dividend_yield=0.0,
        dividend_amount=2.0, ex_dividend_date=dt.date(2027, 3, 15),
        nb_steps=200, pruning_threshold=1e-7, option_type="Put", exercise_type="American",
    )
    # Les autres exemples ne changent que quelques champs de la base.
    if name == PRESETS[1]:
        return replace(base, rate=0.03, vol=0.20, dividend_amount=0.0, option_type="Call",
                       exercise_type="European", pruning_threshold=0.0)
    if name == PRESETS[2]:
        return replace(base, rate=-0.01, dividend_amount=0.0, option_type="Call")
    return base


# Mettre des valeurs dans les widgets AVANT leur création au rerun
def _populate_form(params: Parameters) -> None:
    """Mettre des valeurs dans les widgets AVANT leur création au rerun."""
    for name in ("spot", "strike", "nb_steps", "pruning_threshold", "pricing_date",
                 "maturity"):
        st.session_state["param_" + name] = getattr(params, name)
    # Taux et volatilité sont saisis en pourcentage à l'écran.
    for name in ("rate", "vol", "dividend_yield"):
        st.session_state["param_" + name] = 100.0 * getattr(params, name)
    st.session_state["param_dividend_amount"] = params.dividend_amount
    st.session_state["param_cash_enabled"] = params.dividend_amount > 0
    st.session_state["param_ex_dividend_date"] = params.ex_dividend_date or params.pricing_date
    st.session_state["param_option_type"] = params.option_type
    st.session_state["param_exercise_type"] = (
        "Américaine" if params.exercise_type == "American" else "Européenne")


# Rappel du menu « Exemple de départ » : recharger tout le formulaire
def _change_preset() -> None:
    """Rappel du menu « Exemple de départ » : recharger tout le formulaire."""
    _populate_form(_preset(st.session_state["preset_name"]))


# Champs du marché : spot, taux, volatilité, rendement continu
def _market_inputs() -> None:
    """Champs du marché : spot, taux, volatilité, rendement continu."""
    with st.expander("Marché", expanded=True):
        st.number_input("Spot S₀ (€)", min_value=0.01, key="param_spot", format="%.2f")
        st.number_input("Taux annuel r (%)", min_value=-100.0, max_value=100.0,
                        key="param_rate", step=0.5, format="%.2f",
                        help="Un taux négatif est accepté. 3 signifie 3 % par an.")
        st.number_input("Volatilité σ (%)", min_value=0.0, max_value=300.0,
                        key="param_vol", step=1.0, format="%.2f")
        st.number_input("Rendement continu q (%)", min_value=0.0, max_value=100.0,
                        key="param_dividend_yield", step=0.5, format="%.2f")


# Champs du contrat : type, exercice, strike et dates
def _option_inputs() -> None:
    """Champs du contrat : type, exercice, strike et dates."""
    with st.expander("Option & dates", expanded=True):
        st.selectbox("Type d’option", ("Call", "Put"), key="param_option_type",
                     help="Call : droit d’acheter. Put : droit de vendre.")
        st.selectbox("Exercice", ("Européenne", "Américaine"), key="param_exercise_type")
        st.number_input("Strike K (€)", min_value=0.0, key="param_strike", format="%.2f")
        st.date_input("Date de calcul", key="param_pricing_date", min_value=DATE_MIN,
                      max_value=DATE_MAX, format="DD/MM/YYYY")
        st.date_input("Maturité", key="param_maturity", min_value=DATE_MIN,
                      max_value=DATE_MAX, format="DD/MM/YYYY")


# Champs du dividende cash puis du maillage (N et pruning)
def _cash_and_grid_inputs() -> None:
    """Champs du dividende cash puis du maillage (N et pruning)."""
    with st.expander("Dividende cash", expanded=False):
        st.checkbox("Activer le dividende cash", key="param_cash_enabled")
        st.number_input("Montant D (€)", min_value=0.0, key="param_dividend_amount",
                        step=0.5, format="%.2f")
        st.date_input("Date de détachement", key="param_ex_dividend_date",
                      min_value=DATE_MIN, max_value=DATE_MAX, format="DD/MM/YYYY")
        st.caption("Un seul cash est géré. Le rendement continu q peut aussi être actif.")
    # Le maillage détermine la précision et le temps de calcul.
    with st.expander("Maillage & pruning", expanded=False):
        st.number_input("Nombre de pas N", min_value=1, max_value=2000,
                        key="param_nb_steps", step=10)
        st.number_input("Seuil de pruning", min_value=0.0, max_value=1.0,
                        key="param_pruning_threshold", step=1e-7, format="%.8g",
                        help="0 désactive l’élagage. Exemple : 0.0000001 = 1e-7.")
        st.caption("Les grands N demandent plus de temps et de mémoire.")


# Lire les widgets et convertir les pourcentages en décimaux
def _form_candidate() -> Parameters:
    """Lire les widgets et convertir les pourcentages en décimaux."""
    state = st.session_state
    cash = state["param_cash_enabled"]
    return Parameters(
        pricing_date=state["param_pricing_date"], maturity=state["param_maturity"],
        spot=float(state["param_spot"]), strike=float(state["param_strike"]),
        rate=state["param_rate"] / 100.0, vol=state["param_vol"] / 100.0,
        dividend_yield=state["param_dividend_yield"] / 100.0,
        # Un cash désactivé est transmis comme un montant nul, sans date.
        dividend_amount=float(state["param_dividend_amount"]) if cash else 0.0,
        ex_dividend_date=state["param_ex_dividend_date"] if cash else None,
        nb_steps=int(state["param_nb_steps"]),
        pruning_threshold=float(state["param_pruning_threshold"]),
        option_type=state["param_option_type"],
        exercise_type=("American" if state["param_exercise_type"] == "Américaine"
                       else "European"),
    )


# Valider le scénario saisi ; ne remplacer l'actif que s'il est valide
def _apply_form() -> None:
    """Valider le scénario saisi ; ne remplacer l'actif que s'il est valide."""
    candidate = _form_candidate()
    try:
        candidate.validate()
        st.session_state["active_parameters"] = candidate
        st.success("Scénario appliqué.")
    except ValueError as exc:
        st.error(str(exc))


# Un formulaire évite de reconstruire l'arbre à chaque chiffre tapé
def _sidebar() -> tuple[Parameters, str]:
    """Un formulaire évite de reconstruire l'arbre à chaque chiffre tapé."""
    if "active_parameters" not in st.session_state:
        st.session_state["active_parameters"] = _preset(PRESETS[0])
        _populate_form(st.session_state["active_parameters"])
    with st.sidebar:
        st.markdown("### Arbres d’options")
        st.caption("Laboratoire de valorisation · M2 Dauphine")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.selectbox("Exemple de départ", PRESETS, key="preset_name",
                     on_change=_change_preset)
        st.caption("Choisis un exemple, ajuste les champs, puis applique le scénario.")
        # Les champs du formulaire ne sont lus qu'au clic sur le bouton.
        with st.form("scenario_form"):
            _market_inputs()
            _option_inputs()
            _cash_and_grid_inputs()
            apply = st.form_submit_button("Appliquer le scénario", type="primary",
                                          width="stretch")
        if apply:
            _apply_form()
        active = st.session_state["active_parameters"]
        _sidebar_footer(active)
    return active, page


# Export JSON du scénario actif et lien vers le code
def _sidebar_footer(active: Parameters) -> None:
    """Export JSON du scénario actif et lien vers le code."""
    st.download_button("Exporter les paramètres JSON", scenario_json(active),
                       "parametres.json", "application/json", width="stretch")
    st.link_button("Code et documentation", GITHUB, width="stretch")


# =============================================================================
# 2. OUTILS D'AFFICHAGE COMMUNS
# =============================================================================
@st.cache_data(max_entries=5, ttl=1800, show_spinner=False)
def _cached_summary(params: Parameters) -> dict:
    """Mémoriser des nombres/DataFrames, jamais un Tree et ses cycles."""
    return calculate_summary(params)


# Mémoriser le ZIP des profils (octets seulement)
@st.cache_data(max_entries=3, ttl=1800, show_spinner=False)
def _cached_archive(params: Parameters) -> bytes:
    """Mémoriser le ZIP des profils (octets seulement)."""
    return profile_archive(params)


# Même présentation et export vectoriel pour tous les graphiques
def _plot(fig: go.Figure, key: str) -> None:
    """Même présentation et export vectoriel pour tous les graphiques."""
    fig.update_layout(
        template="plotly_white", colorway=COLORS,
        margin=dict(l=15, r=15, t=45, b=20), hovermode="x unified",
        font=dict(family="Arial, sans-serif", color="#203047"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    # Le bouton appareil photo de Plotly exporte en SVG.
    st.plotly_chart(fig, key=key, width="stretch",
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})


# Une courbe par colonne de ``series`` ({colonne: légende})
def _line_figure(frame: pd.DataFrame, x: str, series: dict[str, str], xlabel: str,
                 ylabel: str, probability: bool = False) -> go.Figure:
    """Une courbe par colonne de ``series`` ({colonne: légende})."""
    fig = go.Figure()
    for column, label in series.items():
        fig.add_trace(go.Scatter(x=frame[x], y=frame[column], name=label,
                                 mode="lines+markers", marker=dict(size=4),
                                 line=dict(width=2)))
    fig.update_xaxes(title=xlabel)
    # Axe en pourcentage pour les probabilités.
    fig.update_yaxes(title=ylabel, tickformat=".0%" if probability else None)
    return fig


# Bouton de téléchargement CSV d'un tableau
def _csv_download(frame: pd.DataFrame, filename: str, key: str) -> None:
    """Bouton de téléchargement CSV d'un tableau."""
    # UTF-8 avec BOM facilite l'ouverture des accents dans Excel sous Windows.
    st.download_button("Télécharger les données CSV",
                       frame.to_csv(index=False).encode("utf-8-sig"),
                       filename, "text/csv", key=key)


# Message d'erreur lisible au lieu d'une trace Python
def _show_errors(exc: Exception) -> None:
    """Message d'erreur lisible au lieu d'une trace Python."""
    st.error("Ce calcul n’a pas pu être effectué avec les paramètres choisis.")
    st.info(str(exc))


# Ne montrer que les résultats correspondant aux réglages affichés
def _study_result(key: str, signature: tuple, compute: Callable[[], object],
                  label: str) -> object | None:
    """Ne montrer que les résultats correspondant aux réglages affichés.

    Les téléchargements et le zoom provoquent des reruns Streamlit :
    session_state garde donc le dernier résultat et sa « signature » (les
    réglages utilisés). Une étude coûteuse n'est lancée que par son bouton.
    """
    if st.button(label, type="primary", key=key + "_button"):
        try:
            with st.spinner("Calcul en cours…"):
                value = compute()
            st.session_state[key] = (signature, value)
        except CALCULATION_ERRORS as exc:
            st.session_state.pop(key, None)
            _show_errors(exc)
    # Un ancien résultat n'est affiché que si les réglages n'ont pas changé.
    saved = st.session_state.get(key)
    if saved is not None and saved[0] == signature:
        return saved[1]
    st.caption("Lance le calcul pour ces réglages. "
               "Les études ne se recalculent pas à chaque saisie.")
    return None


# Conserver les échecs dans le CSV et les rendre visibles, sans prix fictif
def _valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Conserver les échecs dans le CSV et les rendre visibles, sans prix fictif."""
    if "status" not in frame:
        return frame
    failed = frame[frame["status"] != "ok"]
    if not failed.empty:
        st.warning(f"{len(failed)} scénario(s) non représentable(s) sur ce maillage.")
        with st.expander("Voir les scénarios et leurs messages"):
            st.dataframe(failed, hide_index=True, width="stretch")
    return frame[frame["status"] == "ok"]


# Lire une liste de nombres séparés par virgules, points-virgules ou espaces
def _number_list(text: str, integers: bool = False, maximum: int = 20) -> list:
    """Lire une liste de nombres séparés par virgules, points-virgules ou espaces."""
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not 1 <= len(tokens) <= maximum:
        raise ValueError(f"Saisir entre 1 et {maximum} valeurs, séparées par des virgules.")
    try:
        values = [int(token) if integers else float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Utiliser des nombres, avec un point décimal et sans "
                         "symbole %.") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Les valeurs doivent être finies.")
    # Supprimer les doublons en gardant l'ordre de saisie.
    return list(dict.fromkeys(values))


# Synthèse du scénario (en cache) ou None si le calcul échoue
def _summary(params: Parameters) -> dict | None:
    """Synthèse du scénario (en cache) ou None si le calcul échoue."""
    try:
        with st.spinner("Valorisation du scénario…"):
            return _cached_summary(params)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        _show_errors(exc)
        return None


# =============================================================================
# 3. PAGE « PRIX & COMPARAISON »
# =============================================================================
def _price_metrics(result: dict, params: Parameters) -> None:
    """Quatre indicateurs : prix, prime américaine, nœuds et temps."""
    table = result["comparison"]
    same_kind = table[table["option_type"] == params.option_type].set_index(
        "exercise_type")["price"]
    premium = float(same_kind["American"] - same_kind["European"])
    cols = st.columns(4)
    cols[0].metric("Prix du contrat choisi", f"{result['price']:.6f} €")
    cols[1].metric("Prime américaine", f"{premium:.6f} €")
    cols[2].metric("Nœuds construits", f"{result['node_count']:,}".replace(",", " "))
    cols[3].metric("Temps de la synthèse", f"{result['elapsed_seconds']:.3f} s")
    st.caption("La prime compare américain − européen pour le call ou le put sélectionné. "
               "Le temps inclut la construction, les quatre valorisations et le profil.")


# Barres des quatre prix ; le contrat choisi est mis en couleur
def _price_chart(table: pd.DataFrame, params: Parameters) -> None:
    """Barres des quatre prix ; le contrat choisi est mis en couleur."""
    colors = [COLORS[0] if row.option_type == params.option_type
              and row.exercise_type == params.exercise_type else "#cbd5e1"
              for row in table.itertuples()]
    fig = go.Figure(go.Bar(x=table["Contrat"], y=table["price"], marker_color=colors,
                           text=table["price"].map("{:.4f} €".format),
                           textposition="outside"))
    fig.update_yaxes(title="Prix (€)", rangemode="tozero")
    _plot(fig, "comparison_chart")


# Prix du contrat, comparaison des quatre contrats et Black-Scholes
def _price_page(params: Parameters) -> None:
    """Prix du contrat, comparaison des quatre contrats et Black-Scholes."""
    st.subheader("Un scénario, quatre contrats")
    st.write("Compare calls et puts, européens et américains, sur le même sous-jacent.")
    result = _summary(params)
    if result is None:
        return
    _price_metrics(result, params)
    table = result["comparison"].copy()
    table["Contrat"] = [kind + " " + EXERCISE_LABELS[exercise] for kind, exercise
                        in zip(table["option_type"], table["exercise_type"])]
    _price_chart(table, params)
    # Black-Scholes n'est affiché que lorsqu'il est une référence exacte.
    if result["black_scholes"] is not None:
        black = result["black_scholes"]
        st.info(f"Black-Scholes européen : {black:.8f} € · Écart arbre − Black-Scholes : "
                f"{result['price'] - black:+.3g} €.")
    else:
        st.caption(result["black_scholes_reason"])
    st.dataframe(table[["Contrat", "price"]].rename(columns={"price": "Prix (€)"}),
                 hide_index=True, width="stretch")
    _csv_download(result["comparison"], "comparaison_contrats.csv", "comparison_csv")


# =============================================================================
# 4. PAGE « VALEUR & EXERCICE » 
# =============================================================================
def _value_tab(frame: pd.DataFrame, result: dict) -> None:
    """Moyenne actualisée par date et contrôle de conservation de la valeur."""
    st.write("La courbe demandée dans le PDF est la moyenne actualisée de la valeur "
             "sur tous les nœuds de chaque date.")
    fig = _line_figure(frame, "time_years", {"discounted_mean_value": "Moyenne du PDF"},
                       "Temps depuis la date de calcul (années)", "Valeur actualisée (€)")
    fig.add_hline(y=result["price"], line_dash="dash", line_color="#64748b",
                  annotation_text="Prix initial")
    _plot(fig, "mean_value_chart")
    st.caption("Européenne : moyenne constante. Américaine : moyenne non croissante, "
               "car ce premier tracé ignore les paiements déjà intervenus.")
    # Valeur vivante + paiements antérieurs = prix initial à toute date.
    conservation = frame.assign(past_cashflows=(frame["cumulative_discounted_cashflow"]
                                                - frame["discounted_exercise_cashflow"]))
    _plot(_line_figure(conservation, "time_years", {
        "discounted_alive_value": "Contrats vivants, avant décision",
        "past_cashflows": "Paiements strictement antérieurs",
        "total_value": "Somme conservée",
    }, "Temps (années)", "Valeur actualisée (€)"), "conservation_chart")
    st.caption(f"Écart maximal de conservation : {result['value_error']:.3g} € · "
               f"Écart maximal de masse : {result['mass_error']:.3g}.")


# Masse des nœuds en zone d'exercice, date par date
def _region_chart(early: pd.DataFrame, terminal: pd.Series) -> None:
    """Masse des nœuds en zone d'exercice, date par date."""
    fig = _line_figure(early, "time_years", {
        "exercise_region_probability": "Zone d’exercice anticipé",
    }, "Temps (années)", "Probabilité", True)
    # À maturité, l'exercice signifie simplement un payoff positif.
    fig.add_trace(go.Scatter(x=[terminal.time_years],
                             y=[terminal.exercise_region_probability],
                             name="À maturité : payoff positif", mode="markers",
                             marker=dict(size=10, symbol="diamond")))
    _plot(fig, "region_chart")
    st.caption("Ne pas sommer cette courbe dans le temps : "
               "un chemin peut contribuer à plusieurs dates.")


# Distribution du premier exercice (chaque contrat compté une fois)
def _first_exercise_chart(early: pd.DataFrame, terminal: pd.Series) -> None:
    """Distribution du premier exercice (chaque contrat compté une fois)."""
    fig = go.Figure(go.Bar(x=early.time_years, y=early.first_exercise_probability,
                           name="Premier exercice anticipé", marker_color=COLORS[2]))
    fig.add_trace(go.Scatter(x=[terminal.time_years],
                             y=[terminal.first_exercise_probability],
                             mode="markers", name="Premier exercice à maturité",
                             marker=dict(size=11, symbol="diamond", color=COLORS[3])))
    fig.update_xaxes(title="Temps (années)")
    fig.update_yaxes(title="Probabilité par pas", tickformat=".1%")
    _plot(fig, "first_exercise_chart")


# Exercice cumulé, survie avant T et expiration sans paiement à T
def _cumulative_chart(frame: pd.DataFrame, early: pd.DataFrame,
                      terminal: pd.Series) -> None:
    """Exercice cumulé, survie avant T et expiration sans paiement à T."""
    fig = _line_figure(frame, "time_years", {
        "cumulative_exercise_probability": "Exercice cumulé",
    }, "Temps (années)", "Probabilité", True)
    fig.add_trace(go.Scatter(x=early.time_years, y=early.survival_probability,
                             name="Encore vivant après décision, avant T", mode="lines"))
    fig.add_trace(go.Scatter(x=[terminal.time_years], y=[terminal.survival_probability],
                             name="À T : expiration sans paiement", mode="markers",
                             marker=dict(size=11, symbol="x", color="#dc2626")))
    fig.update_yaxes(range=[0, 1.03])
    _plot(fig, "cumulative_chart")


# Trois vues de l'exercice : zone, premier exercice et cumul
def _exercise_tab(frame: pd.DataFrame) -> None:
    """Trois vues de l'exercice : zone, premier exercice et cumul."""
    st.write("La zone d’exercice décrit l’arbre entier. Pour compter chaque contrat "
             "une seule fois, utilise la distribution du premier exercice.")
    # La dernière ligne (maturité) est tracée à part.
    early, terminal = frame.iloc[:-1], frame.iloc[-1]
    _region_chart(early, terminal)
    _first_exercise_chart(early, terminal)
    _cumulative_chart(frame, early, terminal)


# Une courbe par contrat comparé, en fonction de la date de l'arbre
def _scenario_figure(frame: pd.DataFrame, column: str, ylabel: str,
                     probability: bool) -> go.Figure:
    """Une courbe par contrat comparé, en fonction de la date de l'arbre."""
    fig = go.Figure()
    for name, group in frame.groupby("scenario", sort=False):
        fig.add_trace(go.Scatter(x=group["time_years"], y=group[column], name=name,
                                 mode="lines+markers", marker=dict(size=5)))
    fig.update_xaxes(title="Temps depuis la date de calcul (années)")
    fig.update_yaxes(title=ylabel, tickformat=".0%" if probability else None)
    return fig


# Graphique du cours p. 23 : probabilité d'exercice par date et par contrat
def _comparison_tab(params: Parameters) -> None:
    """Graphique : probabilité d'exercice par date et par contrat."""
    st.write(f"Pour chaque date, somme des probabilités des nœuds où le {params.option_type} "
             "est exercé. Les contrats « sans dividende » ont le même forward "
             "à maturité : leur spot est ajusté, comme dans le cours.")
    keys = st.multiselect("Contrats comparés", list(EXERCISE_SCENARIOS),
                          default=list(DEFAULT_SCENARIOS), format_func=EXERCISE_SCENARIOS.get,
                          key="comparison_keys")
    result = _study_result("comparison_result", (params, tuple(keys)),
                           partial(exercise_comparison, params, keys),
                           "Comparer les probabilités d’exercice")
    if result is None:
        return
    _plot(_scenario_figure(result, "exercise_region_probability",
                           "Probabilité d’exercice", True), "comparison_exercise")
    st.caption("Une européenne n’exerce qu’à maturité. Sans dividende, l’américaine "
               "exerce plus tôt et plus souvent ; un dividende retarde l’exercice du put.")
    # Seconde figure du cours : valeur moyenne actualisée par date.
    _plot(_scenario_figure(result, "discounted_mean_value",
                           "Valeur moyenne actualisée (€)", False), "comparison_value")
    prices = result.groupby("scenario", sort=False)["price"].first().reset_index()
    st.dataframe(prices.rename(columns={"scenario": "Contrat", "price": "Prix (€)"}),
                 hide_index=True, width="stretch")
    _csv_download(result, "comparaison_exercice.csv", "comparison_exercise_csv")


# Tableau complet du profil, CSV et ZIP des graphiques
def _data_tab(frame: pd.DataFrame, params: Parameters) -> None:
    """Tableau complet du profil, CSV et ZIP des graphiques."""
    st.dataframe(frame, hide_index=True, width="stretch")
    _csv_download(frame, "profils_option.csv", "profile_csv")
    archive = _study_result("profile_archive", (params,), partial(_cached_archive, params),
                            "Préparer les graphiques PNG/SVG")
    if archive is not None:
        st.download_button("Télécharger les graphiques et données ZIP", archive,
                           "profils_option.zip", "application/zip")


# Valeur moyenne, probabilités d'exercice et comparaison des contrats
def _profile_page(params: Parameters) -> None:
    """Valeur moyenne, probabilités d'exercice et comparaison des contrats."""
    st.subheader("Valeur et exercice, date par date")
    st.caption("Les probabilités sont risque-neutres : elles servent à valoriser le contrat.")
    result = _summary(params)
    if result is None:
        return
    first, second, third = st.columns(3)
    first.metric("Premier exercice avant maturité",
                 f"{result['early_exercise_probability']:.2%}")
    second.metric("Premier exercice à maturité",
                  f"{result['maturity_exercise_probability']:.2%}")
    third.metric("Expiration sans paiement", f"{result['expiration_probability']:.2%}")
    # Un onglet par thème ; la comparaison n'est calculée que sur demande.
    values, exercises, compare, data = st.tabs((
        "Valeur moyenne", "Exercice", "Comparer les contrats", "Données & exports"))
    with values:
        _value_tab(result["profile"], result)
    with exercises:
        _exercise_tab(result["profile"])
    with compare:
        _comparison_tab(params)
    with data:
        _data_tab(result["profile"], params)


# =============================================================================
# 5. PAGE « ARBRE & DIAGNOSTICS »
# =============================================================================
def _svg_view(svg: str, zoom: float) -> None:
    """Redimensionner le canevas SVG en gardant son viewBox vectoriel intact.

    Le dessin peut être plus large que l'écran : l'iframe fournit des barres
    de défilement et le curseur change seulement l'échelle d'affichage.
    """
    root = ET.fromstring(svg)
    root.set("width", str(float(root.get("width", "1000")) * zoom))
    root.set("height", str(float(root.get("height", "600")) * zoom))
    # Préserver le namespace SVG par défaut lors de la sérialisation ElementTree.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    markup = ET.tostring(root, encoding="unicode")
    html = ('<style>body{margin:0;background:#f8fafc}.viewport{overflow:auto;height:690px;'
            'border:1px solid #dbe3ee;border-radius:10px}</style>'
            '<div class="viewport">' + markup + '</div>')
    # API récente, avec repli pour les versions 1.51 antérieures à st.iframe.
    # Le HTML provient exclusivement du débogueur et des nombres validés.
    if hasattr(st, "iframe"):
        st.iframe(html, height=710, alt="Arbre SVG avec nœuds et branches")
    else:
        components.html(html, height=710, scrolling=False)


# Réglages propres au dessin : pas, seuil, couleur et options
def _debug_controls() -> tuple[int, float, str, bool, bool]:
    """Réglages propres au dessin : pas, seuil, couleur et options."""
    a, b, c = st.columns(3)
    steps = int(a.number_input("Pas du dessin", 1, 40, 16, key="debug_steps"))
    threshold = float(b.number_input("Seuil du dessin", 0.0, 1.0, 1e-3,
                                     format="%.8g", key="debug_threshold"))
    colors = {"Valeur de l’option": "option", "Sous-jacent": "spot",
              "Probabilité": "probability"}
    color = colors[c.selectbox("Couleur des nœuds", tuple(colors))]
    neighbors = st.checkbox("Afficher les liens entre voisins", value=False)
    compare = st.checkbox("Comparer avec l’arbre sans pruning", value=True)
    return steps, threshold, color, neighbors, compare


# Dessin demandé et, si besoin, le même arbre sans pruning
def _debug_trees(params: Parameters, steps: int, color: str, neighbors: bool,
                 compare: bool) -> dict:
    """Dessin demandé et, si besoin, le même arbre sans pruning."""
    results = {"Dessin choisi": debug_tree(params, steps, color, neighbors)}
    if compare and params.pruning_threshold > 0:
        results["Sans pruning"] = debug_tree(replace(params, pruning_threshold=0.0),
                                             steps, color, neighbors)
    return results


# Indicateurs, dessin SVG et rapport d'un arbre contrôlé
def _show_diagnostic(name: str, diagnostic: dict, zoom: float) -> None:
    """Indicateurs, dessin SVG et rapport d'un arbre contrôlé."""
    cols = st.columns(3)
    cols[0].metric("Nœuds", diagnostic["node_count"])
    cols[1].metric("Transitions élaguées", diagnostic["pruned_count"])
    cols[2].metric("Erreurs détectées", diagnostic["errors"])
    # Pas de dessin si les contrôles ont trouvé une incohérence.
    if diagnostic["svg"]:
        _svg_view(diagnostic["svg"], zoom / 100.0)
        filename = "arbre_" + ("complet" if name == "Sans pruning" else "choisi") + ".svg"
        st.download_button("Télécharger le SVG complet", diagnostic["svg"], filename,
                           "image/svg+xml", key="svg_" + name)
    else:
        st.error("Le dessin est suspendu : les contrôles ont trouvé une incohérence. "
                 "Voir le rapport.")
    with st.expander("Rapport des contrôles"):
        st.code(diagnostic["report"], language="text")


# Contrôler le scénario principal sans dessiner tous ses nœuds
def _main_check(params: Parameters) -> None:
    """Contrôler le scénario principal sans dessiner tous ses nœuds."""
    st.write(f"Contrôler aussi le scénario principal à N = {params.nb_steps}, "
             "sans dessiner tous les nœuds.")
    check = _study_result("main_check", (params,), partial(check_tree, params),
                          "Contrôler le scénario principal")
    if check is None:
        return
    if check["errors"] == 0:
        st.success(f"Aucune erreur de construction détectée sur {check['node_count']} nœuds.")
    else:
        st.error(f"{check['errors']} erreur(s) détectée(s).")
    st.code(check["report"], language="text")


# Dessin SVG d'un petit arbre et contrôles numériques
def _debug_page(params: Parameters) -> None:
    """Dessin SVG d'un petit arbre et contrôles numériques."""
    st.subheader("Inspecter la construction de l’arbre")
    st.write("Survole les nœuds pour lire le payoff, la continuation et les moments. "
             "Les branches orange signalent une redirection de pruning.")
    steps, threshold, color, neighbors, compare = _debug_controls()
    st.caption("Le nombre de pas et le seuil ci-dessus sont propres au dessin. "
               "Le marché, l’option et les dates restent ceux du scénario appliqué.")
    drawing = replace(params, pruning_threshold=threshold)
    result = _study_result("debug_result", (drawing, steps, color, neighbors, compare),
                           partial(_debug_trees, drawing, steps, color, neighbors, compare),
                           "Générer et contrôler les arbres")
    # Un onglet par arbre dessiné (choisi, et éventuellement sans pruning).
    if result is not None:
        zoom = st.slider("Zoom du dessin (%)", 10, 150, 35, step=5)
        for tab, (name, diagnostic) in zip(st.tabs(list(result)), result.items()):
            with tab:
                _show_diagnostic(name, diagnostic, zoom)
    st.divider()
    _main_check(params)


# =============================================================================
# 6. PAGE « CONVERGENCE »
# =============================================================================
def _reference_params(params: Parameters) -> Parameters | None:
    """Une comparaison Black-Scholes doit expliciter tout changement de contrat."""
    st.caption("Comparaison européenne, avec rendement continu q ; "
               "pruning désactivé pour isoler N.")
    if not has_active_cash(params):
        return params
    ignore_cash = st.checkbox("Faire cette étude sans le dividende cash", value=False,
                              key="convergence_ignore_cash")
    if not ignore_cash:
        st.info("Black-Scholes n’est pas une référence exacte avec un cash actif. "
                "Coche la case pour étudier explicitement le scénario sans cash.")
        return None
    return replace(params, dividend_amount=0.0, ex_dividend_date=None)


# Lire la liste des N puis lancer l'étude de convergence
def _convergence_steps(reference: Parameters, text: str) -> pd.DataFrame:
    """Lire la liste des N puis lancer l'étude de convergence."""
    return convergence_study(reference, _number_list(text, True))


# Temps observé en log-log, comparé à une pente N²
def _convergence_time_chart(good: pd.DataFrame) -> None:
    """Temps observé en log-log, comparé à une pente N²."""
    fig = _line_figure(good, "nb_steps", {"seconds": "Temps observé"}, "N", "Secondes")
    last = good.iloc[-1]
    fig.add_trace(go.Scatter(x=good.nb_steps,
                             y=last.seconds * (good.nb_steps / last.nb_steps) ** 2,
                             name="Repère N²", line=dict(dash="dash")))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    _plot(fig, "convergence_time")


# Prix, écart brut, écart × N et temps en fonction de N
def _convergence_charts(good: pd.DataFrame) -> None:
    """Prix, écart brut, écart × N et temps en fonction de N."""
    good = good.sort_values("nb_steps")
    _plot(_line_figure(good, "nb_steps", {"price": "Arbre", "reference": "Black-Scholes"},
                       "Nombre de pas N", "Prix (€)"), "convergence_prices")
    a, b = st.columns(2)
    # Écart × N borné : l'erreur décroît en 1/N.
    with a:
        _plot(_line_figure(good, "nb_steps", {"error": "Écart brut"}, "N",
                           "Arbre − Black (€)"), "convergence_error")
    with b:
        _plot(_line_figure(good, "nb_steps", {"scaled_error": "Écart × N"}, "N",
                           "Écart × N"), "convergence_scaled")
    _convergence_time_chart(good)


# Étude en fonction du nombre de pas N
def _steps_study(reference: Parameters) -> None:
    """Étude en fonction du nombre de pas N."""
    text = st.text_input("N à comparer (20 valeurs maximum)", "5,10,20,50,100,200,400")
    result = _study_result("convergence_result", (reference, text),
                           partial(_convergence_steps, reference, text),
                           "Calculer la convergence en N")
    if result is None:
        return
    good = _valid_rows(result)
    if not good.empty:
        _convergence_charts(good)
    st.dataframe(result, hide_index=True, width="stretch")
    _csv_download(result, "convergence.csv", "convergence_csv")


# Grille de strikes régulière puis étude pour quelques N
def _strike_compute(reference: Parameters, minimum: float, maximum: float, count: int,
                    text: str) -> pd.DataFrame:
    """Grille de strikes régulière puis étude pour quelques N."""
    if maximum <= minimum:
        raise ValueError("Le strike maximum doit être supérieur au minimum.")
    return strike_study(reference, np.linspace(minimum, maximum, count).tolist(),
                        _number_list(text, True, 4))


# Écart arbre − Black-Scholes selon le strike, une courbe par N
def _strike_chart(result: pd.DataFrame) -> None:
    """Écart arbre − Black-Scholes selon le strike, une courbe par N."""
    fig = go.Figure()
    for n, group in result.groupby("nb_steps"):
        fig.add_trace(go.Scatter(x=group.strike, y=group.error, name=f"N = {n}",
                                 mode="lines"))
    fig.add_hline(y=0, line_color="#94a3b8")
    fig.update_xaxes(title="Strike K (€)")
    fig.update_yaxes(title="Arbre − Black-Scholes (€)")
    _plot(fig, "strike_error")


# Étude en fonction du strike, pour quelques N
def _strike_study(reference: Parameters, params: Parameters) -> None:
    """Étude en fonction du strike, pour quelques N."""
    left, middle, right = st.columns(3)
    minimum = float(left.number_input("Strike minimum", min_value=0.0,
                                      value=max(0.0, params.strike * 0.8)))
    maximum = float(middle.number_input("Strike maximum", min_value=0.0,
                                        value=max(1.0, params.strike * 1.2)))
    count = int(right.number_input("Nombre de strikes", 2, 81, 31))
    text = st.text_input("N à comparer (4 valeurs maximum)", "10,100", key="strike_steps")
    result = _study_result("strike_result", (reference, minimum, maximum, count, text),
                           partial(_strike_compute, reference, minimum, maximum, count, text),
                           "Calculer l’écart selon le strike")
    if result is not None:
        _strike_chart(result)
        st.dataframe(result, hide_index=True, width="stretch")
        _csv_download(result, "convergence_strike.csv", "strike_csv")


# Comparer l'arbre à Black-Scholes selon N ou selon le strike
def _convergence_page(params: Parameters) -> None:
    """Comparer l'arbre à Black-Scholes selon N ou selon le strike."""
    st.subheader("Convergence vers Black-Scholes")
    reference = _reference_params(params)
    if reference is None:
        return
    mode = st.radio("Étude", ("Nombre de pas N", "Strike K"), horizontal=True,
                    key="convergence_mode")
    if mode == "Nombre de pas N":
        _steps_study(reference)
    else:
        _strike_study(reference, params)


# =============================================================================
# 7. PAGE « PRUNING »
# =============================================================================
def _threshold_bar(display: pd.DataFrame, column: str, ylabel: str, color: str,
                   key: str) -> None:
    """Barres d'une mesure en fonction du seuil de pruning."""
    fig = go.Figure(go.Bar(x=display.label, y=display[column], marker_color=color))
    fig.update_xaxes(title="Seuil de pruning", type="category")
    fig.update_yaxes(title=ylabel, tickformat=".2e" if column == "price_gap" else None)
    _plot(fig, key)


# Nœuds, temps et écart de prix pour chaque seuil
def _pruning_charts(good: pd.DataFrame) -> None:
    """Nœuds, temps et écart de prix pour chaque seuil."""
    display = good.assign(label=good.threshold.map("{:g}".format))
    a, b = st.columns(2)
    with a:
        _threshold_bar(display, "node_count", "Nombre de nœuds", COLORS[0], "pruning_nodes")
    with b:
        _threshold_bar(display, "seconds", "Construction + pricing (s)", COLORS[1],
                       "pruning_times")
    _threshold_bar(display, "price_gap", "Écart de prix au seuil 0 (€)", COLORS[2],
                   "pruning_gap")


# Lire la liste des seuils puis lancer la comparaison
def _pruning_thresholds(target: Parameters, text: str, steps: int) -> pd.DataFrame:
    """Lire la liste des seuils puis lancer la comparaison."""
    return pruning_study(target, _number_list(text), steps)


# Comparer plusieurs seuils de pruning au même nombre de pas
def _pruning_page(params: Parameters) -> None:
    """Comparer plusieurs seuils de pruning au même nombre de pas."""
    st.subheader("Mesurer l’effet du pruning")
    st.write("Compare les seuils au même nombre de pas. "
             "Le seuil 0 fournit la référence sans élagage.")
    steps = int(st.number_input("N de la comparaison", 1, 2000, min(params.nb_steps, 1000),
                                key="pruning_steps"))
    text = st.text_input("Seuils à comparer", "0,1e-7,1e-5,1e-3")
    without_cash = st.checkbox("Faire cette comparaison sans cash", value=False,
                               key="pruning_without_cash")
    target = (replace(params, dividend_amount=0.0, ex_dividend_date=None)
              if without_cash else params)
    result = _study_result("pruning_result", (target, text, steps),
                           partial(_pruning_thresholds, target, text, steps),
                           "Comparer les seuils")
    if result is None:
        return
    good = _valid_rows(result)
    if not good.empty:
        _pruning_charts(good)
    # Le tableau complet garde aussi les seuils en échec.
    st.caption("Un seuil sur la probabilité ne garantit pas une erreur de prix maximale. "
               "Si l’arbre complet échoue, aucun écart de référence n’est inventé. "
               "Les temps dépendent de la machine.")
    st.dataframe(result, hide_index=True, width="stretch")
    _csv_download(result, "comparaison_pruning.csv", "pruning_csv")


# =============================================================================
# 8. PAGE « DIVIDENDES & TAUX »
# =============================================================================
def _four_price_chart(frame: pd.DataFrame, x: str, xlabel: str, key: str) -> None:
    """Prix des quatre contrats en fonction du paramètre étudié."""
    good = _valid_rows(frame)
    if good.empty:
        return
    _plot(_line_figure(good, x, {
        "call_european": "Call européen", "call_american": "Call américain",
        "put_european": "Put européen", "put_american": "Put américain",
    }, xlabel, "Prix (€)"), key)


# Lire les montants cash puis lancer l'étude
def _cash_amounts(target: Parameters, text: str) -> pd.DataFrame:
    """Lire les montants cash puis lancer l'étude."""
    return dividend_study(target, _number_list(text, False, 15))


# Faire varier le dividende cash D (date choisie pour l'étude)
def _cash_study(target: Parameters, params: Parameters) -> tuple[object | None, str, str]:
    """Faire varier le dividende cash D (date choisie pour l'étude)."""
    text = st.text_input("Montants cash en euros", "0,1,2,3,4")
    # Même si le cash du formulaire est désactivé, l'étude doit
    # disposer d'une date explicite pour ses montants positifs.
    middle = params.pricing_date + (params.maturity - params.pricing_date) // 2
    date = st.date_input("Détachement de l’étude", value=params.ex_dividend_date or middle,
                         min_value=DATE_MIN, max_value=DATE_MAX, format="DD/MM/YYYY")
    target = replace(target, ex_dividend_date=date)
    st.caption(f"q reste fixé à {target.dividend_yield:.2%}. Le montant 0 signifie sans cash.")
    if not target.pricing_date < date <= target.maturity:
        st.info("Ce détachement est hors de la vie du contrat : "
                "les montants cash étudiés seront sans effet.")
    result = _study_result("dividend_result", (target, text),
                           partial(_cash_amounts, target, text),
                           "Comparer les dividendes cash")
    return result, "dividend_amount", "Dividende cash D (€)"


# Une ligne de quatre prix par rendement continu q (en %)
def _yield_compute(target: Parameters, text: str) -> pd.DataFrame:
    """Une ligne de quatre prix par rendement continu q (en %)."""
    rows = []
    for q_percent in _number_list(text, False, 15):
        scenario = replace(target, dividend_yield=q_percent / 100.0)
        row = dividend_study(scenario, [0.0]).iloc[0].to_dict()
        row["yield_percent"] = q_percent
        rows.append(row)
    return pd.DataFrame(rows)


# Faire varier le rendement continu q, sans dividende cash
def _yield_study(target: Parameters) -> tuple[object | None, str, str]:
    """Faire varier le rendement continu q, sans dividende cash."""
    text = st.text_input("Rendements continus en %", "0,1,2,3,4", key="yield_list")
    target = replace(target, dividend_amount=0.0, ex_dividend_date=None)
    st.caption("Cette étude fait varier q, sans dividende cash.")
    result = _study_result("yield_result", (target, text),
                           partial(_yield_compute, target, text),
                           "Comparer les rendements continus")
    return result, "yield_percent", "Rendement continu q (%)"


# Effet du cash D ou du rendement q sur les quatre contrats
def _dividend_tab(params: Parameters) -> None:
    """Effet du cash D ou du rendement q sur les quatre contrats."""
    rate_zero = st.checkbox("Fixer le taux à 0 % pour cette étude (cas du PDF)", value=True)
    target = replace(params, rate=0.0) if rate_zero else params
    mode = st.radio("Paramètre étudié", ("Cash D", "Rendement continu q"), horizontal=True,
                    key="dividend_mode")
    if mode == "Cash D":
        result, x, xlabel = _cash_study(target, params)
    else:
        result, x, xlabel = _yield_study(target)
    if result is not None:
        _four_price_chart(result, x, xlabel, "dividend_chart")
        st.dataframe(result, hide_index=True, width="stretch")
        _csv_download(result, "impact_dividendes.csv", "dividend_csv")


# Convertir les taux saisis en % puis lancer l'étude
def _rate_values(params: Parameters, text: str) -> pd.DataFrame:
    """Convertir les taux saisis en % puis lancer l'étude."""
    return rate_study(params, [value / 100.0 for value in _number_list(text, False, 15)])


# Prix et primes américaines pour des taux positifs et négatifs
def _rate_tab(params: Parameters) -> None:
    """Prix et primes américaines pour des taux positifs et négatifs."""
    text = st.text_input("Taux annuels à comparer, en %", "-3,-1,0,1,3,5,7")
    result = _study_result("rate_result", (params, text), partial(_rate_values, params, text),
                           "Comparer les taux et les primes américaines")
    if result is None:
        return
    frame = result.assign(rate_percent=result.rate * 100.0)
    _four_price_chart(frame, "rate_percent", "Taux annuel r (%)", "rate_prices")
    good = _valid_rows(frame)
    # Prime américaine = valeur du droit d'exercer avant maturité.
    if not good.empty:
        _plot(_line_figure(good, "rate_percent", {
            "call_premium": "Prime américaine du call", "put_premium": "Prime américaine du put",
        }, "Taux annuel r (%)", "Américain − européen (€)"), "rate_premium")
    st.dataframe(frame, hide_index=True, width="stretch")
    _csv_download(frame, "impact_taux.csv", "rate_csv")


# Deux onglets : dividendes, puis taux
def _dividend_rate_page(params: Parameters) -> None:
    """Deux onglets : dividendes, puis taux."""
    st.subheader("Dividendes, taux et exercice anticipé")
    dividends, rates = st.tabs(("Effet des dividendes", "Américain − européen / taux"))
    with dividends:
        _dividend_tab(params)
    with rates:
        _rate_tab(params)


# =============================================================================
# 8 bis. PAGE « GRECQUES & COUVERTURE » (cours p. 17)
# =============================================================================
# Libellés des colonnes du tableau des grecques
GREEK_COLUMNS = {
    "greek": "Grecque", "tree": "Arbre (différences finies)",
    "tree_nodes": "Arbre (nœuds voisins)",
    "tree_control_variate": "Arbre + variable de contrôle",
    "black_scholes": "Black-Scholes", "gap": "Arbre − Black-Scholes", "unit": "Unité",
}
# Libellés des termes du développement de Taylor
TAYLOR_LABELS = {
    "delta": "Δ · dS", "gamma": "½ Γ · dS²", "vega": "vega · dσ", "vomma": "½ vomma · dσ²",
    "vanna": "vanna · dS · dσ", "theta": "θ · jours", "total": "Total (option seule)",
    "hedged_total": "Total (option − Δ actions)", "repriced": "Repricing exact (option seule)",
    "repriced_hedged": "Repricing exact (option − Δ actions)",
}


# Six indicateurs en haut de la page : prix et grecques principales
def _greek_metrics(values: dict) -> None:
    """Prix, delta, gamma, vega, vanna et theta du contrat choisi."""
    cols = st.columns(6)
    cols[0].metric("Prix", f"{values['price']:.4f} €")
    cols[1].metric("Delta Δ", f"{values['delta']:.4f}")
    cols[2].metric("Gamma Γ", f"{values['gamma']:.5f}")
    cols[3].metric("Vega (1 pt)", f"{values['vega']:.4f} €")
    cols[4].metric("Vanna (1 pt)", f"{values['vanna']:.5f}")
    cols[5].metric("Theta (1 jour)", f"{values['theta']:.4f} €")


# Tableau des grecques et explications des méthodes
def _greek_table_view(result: dict) -> None:
    """Tableau comparatif ; les colonnes vides (sans objet) sont masquées."""
    table = result["table"].dropna(axis=1, how="all").rename(columns=GREEK_COLUMNS)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(result["black_reason"] + f" Calcul : {result['seconds']:.2f} s.")
    # Rappel des méthodes de calcul utilisées par le module grecques.py.
    with st.expander("Comment ces grecques sont calculées"):
        st.write("Différences finies centrées (cours p. 17) : chaque prix décalé "
                 "reconstruit un arbre. Le spot est décalé d’un cran de grille "
                 "(S×α et S/α, les nœuds voisins de la racine), la volatilité de "
                 "±1 point ; le theta est la variation du prix quand un jour passe. "
                 "« Nœuds voisins » : delta et gamma lus sur les trois nœuds du pas 2, "
                 "sans nouvel arbre. Pour une américaine sans cash, la variable de "
                 "contrôle corrige l’erreur de l’arbre mesurée sur l’européenne : "
                 "c’est la valeur la plus fiable pour le vomma.")
    _csv_download(result["table"], "grecques.csv", "greeks_csv")


# Quantité de sous-jacent à détenir pour couvrir le contrat
def _hedge_advice(values: dict, params: Parameters) -> None:
    """Couverture delta d'une option vendue (ou achetée)."""
    delta = values["delta"]
    action = "acheter" if delta > 0 else "vendre"
    st.info(f"Couverture delta : pour **1 option vendue**, {action} **{abs(delta):.4f}** "
            f"action(s), soit {abs(delta) * params.spot:.2f} € de sous-jacent. "
            "Pour une option achetée, faire l’inverse. Le gamma indique de combien "
            f"ce delta change si le spot bouge de 1 € ({values['gamma']:.5f}).")


# Calcul du P&L d'un choc (Taylor puis repricing), lancé par un bouton
def _taylor_compute(params: Parameters, values: dict, spot_move: float,
                    vol_move: float, days: int) -> pd.DataFrame:
    """Tableau des termes de Taylor et du repricing exact."""
    result = taylor_check(params, values, spot_move, vol_move, days)
    return pd.DataFrame({"Terme": [TAYLOR_LABELS[key] for key in result],
                         "Variation (€)": list(result.values())})


# Saisie d'un choc de marché et comparaison Taylor / repricing
def _taylor_section(params: Parameters, values: dict) -> None:
    """P&L de l'option et du portefeuille couvert pour un choc donné."""
    st.markdown("#### P&L après un choc de marché")
    left, middle, right = st.columns(3)
    spot_move = float(left.number_input("Variation du spot dS (€)", value=1.0, step=0.5))
    vol_move = float(middle.number_input("Variation de vol dσ (points)", value=0.0,
                                         step=0.5))
    days = int(right.number_input("Jours écoulés", min_value=0, value=1, step=1))
    result = _study_result("taylor_result", (params, values["price"], spot_move, vol_move,
                                             days),
                           partial(_taylor_compute, params, values, spot_move, vol_move, days),
                           "Calculer le P&L du choc")
    if result is not None:
        st.dataframe(result, hide_index=True, width="stretch")
        st.caption("Le portefeuille « option − Δ actions » est couvert en delta : il reste "
                   "surtout le gamma (gain si le spot bouge, pour une option achetée) et "
                   "le theta (coût du temps). L’écart avec le repricing mesure les termes "
                   "d’ordre supérieur.")


# Delta et gamma pour une grille de spots
def _spot_profile_compute(params: Parameters, minimum: float, maximum: float,
                          count: int) -> pd.DataFrame:
    """Grille régulière de spots puis delta/gamma de l'arbre en chaque point."""
    if not 0 < minimum < maximum:
        raise ValueError("Il faut 0 < spot minimum < spot maximum.")
    return greeks_by_spot(params, np.linspace(minimum, maximum, count).tolist())


# Graphiques du delta et du gamma en fonction du spot
def _spot_profile_section(params: Parameters) -> None:
    """Profil du delta et du gamma : utile pour anticiper les réajustements."""
    st.markdown("#### Delta et gamma selon le spot")
    left, middle, right = st.columns(3)
    minimum = float(left.number_input("Spot minimum", min_value=0.01,
                                      value=round(params.spot * 0.7, 2)))
    maximum = float(middle.number_input("Spot maximum", min_value=0.01,
                                        value=round(params.spot * 1.3, 2)))
    count = int(right.number_input("Nombre de spots", 3, 41, 13))
    result = _study_result("spot_profile_result", (params, minimum, maximum, count),
                           partial(_spot_profile_compute, params, minimum, maximum, count),
                           "Tracer delta et gamma")
    if result is None:
        return
    a, b = st.columns(2)
    # Un graphique par grecque, en fonction du spot.
    with a:
        _plot(_line_figure(result, "spot", {"delta": "Delta"}, "Spot S₀ (€)", "Delta"),
              "spot_delta")
    with b:
        _plot(_line_figure(result, "spot", {"gamma": "Gamma"}, "Spot S₀ (€)", "Gamma"),
              "spot_gamma")
    _csv_download(result, "delta_gamma_spot.csv", "spot_profile_csv")


# Page complète : grecques, couverture, choc et profil en spot
def _greeks_page(params: Parameters) -> None:
    """Grecques du contrat choisi et outils de delta hedging."""
    st.subheader("Grecques et couverture delta")
    st.write("Les grecques sont calculées sur l’arbre par différences finies, avec le N et le pruning du scénario.")
    vol_shift = float(st.number_input("Décalage de volatilité (points)", 0.1, 10.0, 1.0,
                                      step=0.5, key="greeks_vol_shift"))
    result = _study_result("greeks_result", (params, vol_shift),
                           partial(greeks_summary, params, vol_shift / 100.0),
                           "Calculer les grecques")
    if result is None:
        return
    _greek_metrics(result["values"])
    _greek_table_view(result)
    _hedge_advice(result["values"], params)
    _taylor_section(params, result["values"])
    _spot_profile_section(params)


# =============================================================================
# 9. PAGE « GUIDE »
# =============================================================================
GUIDE_ROWS = (
    ("Prix & comparaison", "Prix du contrat, quatre prix call/put européen/américain "
                           "et référence Black-Scholes."),
    ("Valeur & exercice", "Moyenne actualisée, premier exercice, cumul, survie, "
                          "conservation de valeur et comparaison des contrats."),
    ("Grecques & couverture", "Delta, gamma, vega, vomma, vanna, theta ; couverture delta "
                              "et P&L estimé après un choc de marché."),
    ("Arbre & diagnostics", "SVG avec zoom, liens, exercice, pruning et contrôles."),
    ("Convergence", "Écart à Black-Scholes selon N ou le strike, ordre d’erreur et temps."),
    ("Pruning", "Prix, nombre de nœuds et durée selon le seuil d’élagage."),
    ("Dividendes & taux", "Effet du cash, du rendement continu et des taux "
                          "positifs/négatifs."),
)
# Textes des sections dépliables : (titre, paragraphe).
GUIDE_SECTIONS = (
    ("Unités et conventions",
     "Les taux et la volatilité sont saisis en pourcentage : 3 signifie 3 %. Les montants "
     "sont en euros. Le temps est calculé en ACT/365. Une américaine peut exercer aux dates "
     "du maillage ; une européenne à maturité seulement. À maturité, un payoff strictement "
     "positif compte comme exercice. Avant maturité, une égalité numérique conduit à "
     "conserver le contrat."),
    ("Dividendes et limites du modèle",
     "Le moteur gère un rendement continu et un seul cash déterministe. Le pas de "
     "détachement utilise trois branches ; les autres pas sont binomiaux. Certains "
     "paramètres cash produisent des probabilités trinômiales négatives : l’écran explique "
     "alors l’échec, il ne force pas un prix."),
    ("Ce que mesure le pruning",
     "Une coupe redirige la masse du bord vers un enfant intérieur avec probabilité 1. "
     "La masse est conservée mais les moments locaux changent. Compare plusieurs seuils "
     "au même N ; aucun seuil ne garantit seul une précision."),
)


# Mode d'emploi de l'application et rappels du modèle
def _guide_page(params: Parameters) -> None:
    """Mode d'emploi de l'application et rappels du modèle."""
    st.subheader("Prendre en main le laboratoire")
    st.write("1. Choisis un exemple dans la colonne de gauche. 2. Modifie ses paramètres et "
             "applique le scénario. 3. Choisis une page et lance les études.")
    st.dataframe(pd.DataFrame(GUIDE_ROWS, columns=["Page", "Ce qu’elle permet d’étudier"]),
                 hide_index=True, width="stretch")
    # Une section dépliable par thème.
    for index, (title, text) in enumerate(GUIDE_SECTIONS):
        with st.expander(title, expanded=index == 0):
            st.write(text)
    with st.expander("Lancer sur Windows / VS Code"):
        st.code("python -m pip install -r requirements.txt\n"
                "python -m streamlit run app.py", language="powershell")
        st.write("Le navigateur ouvre normalement http://localhost:8501. Garde le terminal "
                 "ouvert ; Ctrl+C arrête le serveur.")
    st.caption(f"Scénario actif : N = {params.nb_steps}. Les grecs ne font pas partie "
               "du moteur.")
    st.link_button("Lire le guide détaillé sur GitHub", GITHUB + "#readme")


# =============================================================================
# 10. POINT D'ENTRÉE : style, bandeau et choix de la page
# =============================================================================
PAGE_STYLE = """<style>
.block-container{max-width:1500px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;
  border-radius:12px;padding:16px 18px}
.hero{background:#10223a;color:#fff;padding:28px 32px;border-radius:16px;margin-bottom:18px}
.hero .eyebrow{font-size:12px;letter-spacing:.12em;color:#8ac8ff;text-transform:uppercase}
.hero h1{font-size:32px;line-height:1.2;margin:8px 0 10px;color:#fff}
.hero p{font-size:15px;margin:0;color:#d7e3f1}
</style>"""
HERO = ('<div class="hero"><div class="eyebrow">Binomial · Trinomial · Exercice optimal</div>'
        '<h1>Explore ton arbre d’options</h1><p>Modifie un scénario, observe le prix et '
        'comprends chaque décision d’exercice.</p></div>')


# Rappel compact du scénario appliqué, en haut de chaque page
def _scenario_caption(params: Parameters) -> None:
    """Rappel compact du scénario appliqué, en haut de chaque page."""
    st.caption(f"SCÉNARIO APPLIQUÉ · {params.option_type} "
               f"{EXERCISE_LABELS[params.exercise_type]} · S₀ {params.spot:g} € · "
               f"K {params.strike:g} € · r {params.rate:.2%} · σ {params.vol:.2%} · "
               f"q {params.dividend_yield:.2%} · D {params.dividend_amount:g} € · "
               f"N {params.nb_steps} · {params.pricing_date:%d/%m/%Y} → "
               f"{params.maturity:%d/%m/%Y}")
    if params.dividend_amount > 0 and not has_active_cash(params):
        st.caption("Le dividende cash est hors de la vie du contrat : "
                   "il n’intervient pas dans cette valorisation.")


# Construire la page : style, formulaire, bandeau puis page choisie
def main() -> None:
    """Construire la page : style, formulaire, bandeau puis page choisie."""
    st.set_page_config(page_title="Arbres d’options · Laboratoire", page_icon="📈",
                       layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    params, page = _sidebar()
    st.markdown(HERO, unsafe_allow_html=True)
    _scenario_caption(params)
    # Seule la page choisie est calculée (les autres ne coûtent rien).
    pages = dict(zip(PAGES, (_price_page, _profile_page, _greeks_page, _debug_page,
                             _convergence_page, _pruning_page, _dividend_rate_page,
                             _guide_page)))
    pages[page](params)


if __name__ == "__main__":
    main()
