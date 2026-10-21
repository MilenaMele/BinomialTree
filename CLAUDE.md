# Règles du projet (pricer par arbre, M2 Dauphine)

Le professeur note le code avec un outil automatique. Toute modification doit
respecter les règles ci-dessous, sinon la note baisse
(« long method », « lack of private method », « number of lines by method »).

## Règles de style vérifiées par l'outil du professeur

1. **Méthodes courtes** : au plus **20 instructions** par fonction ou méthode
   (docstring, ligne `def` et lignes de continuation non comptées). Au-delà,
   découper en petites méthodes privées.
2. **Tout est typé** : chaque paramètre (sauf `self`/`cls`) et chaque valeur de
   retour (`-> None` compris), y compris les fonctions internes.
   - Pas de `*` seul ni de `*args`/`**kwargs` dans une signature : l'outil compte
     `*` comme un paramètre non typé.
   - Pas de fonction imbriquée (`def` dans un `def`) : utiliser une fonction de
     module privée et `functools.partial`. Éviter les `lambda`.
3. **Commentaires réguliers** : jamais plus de **20 lignes non vides** sans
   commentaire `#` ou docstring (l'outil tolère 25 ; on garde une marge).
4. **Méthodes privées** : chaque classe a au moins une méthode privée
   (`_nom`). Les fonctions utilisées seulement dans leur fichier sont privées
   (`_nom`) ; seules celles appelées depuis un autre module restent publiques.
5. **Lignes courtes** : au plus **99 caractères** par ligne.
6. **Docstring de module** : chaque `.py` commence par une explication
   « À quoi sert ce module ? ».

Commentaires et docstrings sont rédigés en français.

## Architecture

| Fichier | Rôle |
|---|---|
| `arbre_binomial.py` | Moteur : `Market`, `Option`, `Model`, `Node`, `TrunkNode`, `Tree` (construction, pricing, profils) |
| `profils_option.py` | Figures Matplotlib et CSV des profils par date |
| `debug_arbre.py` | Contrôles numériques de l'arbre et export SVG |
| `convergence.py` | Prix Black-Scholes et étude de convergence |
| `interface_calculs.py` | Pont écran → moteur : validation, études, tableaux pandas (sans Streamlit) |
| `interface_exports.py` | Téléchargements (JSON, ZIP) |
| `app.py` | Seul fichier Streamlit : formulaire, pages, graphiques Plotly |

Contraintes du cours à préserver :

- Aucun conteneur (liste, dictionnaire, tableau) de **nœuds** : on parcourt
  l'arbre uniquement par les liens `next_trunk`, `above`/`below`,
  `next_up`/`next_mid`/`next_down` (cours p. 18).
- Les fonctions de `interface_calculs.py` ne renvoient jamais un `Tree` ni un
  nœud (seulement des nombres et des DataFrames), pour le cache Streamlit.

## Lancer

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
