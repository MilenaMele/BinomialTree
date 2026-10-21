"""Téléchargements de l'interface : scénario JSON et archive ZIP des profils.

À quoi sert ce module ?
    Il prépare les fichiers que l'utilisateur télécharge depuis Streamlit :
    les paramètres du scénario en JSON, et un ZIP contenant les graphiques
    PNG/SVG et le CSV produits par ``profils_option.py``.

Chaque appel possède son propre dossier temporaire, supprimé à la fin :
deux utilisateurs ne peuvent pas écraser les résultats l'un de l'autre.
"""

import gc
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from interface_calculs import Parameters, make_tree
from profils_option import OptionProfiles


def scenario_json(params: Parameters) -> str:
    """Les dates ISO rendent un scénario lisible et reproductible."""
    return json.dumps(asdict(params), default=str, ensure_ascii=False, indent=2)


def _zip_files(files: dict[str, Path], params: Parameters) -> bytes:
    """Mettre les fichiers produits et les paramètres dans un ZIP en mémoire."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in files.values():
            archive.writestr(Path(path).name, Path(path).read_bytes())
        archive.writestr("parametres.json", scenario_json(params))
    return buffer.getvalue()


def profile_archive(params: Parameters) -> bytes:
    """Retourner un ZIP en mémoire : 2 PNG, 2 SVG, données CSV et paramètres.

    Garder les octets du ZIP en cache ne conserve aucun nœud de l'arbre.
    """
    params.validate()
    tree = None
    try:
        tree = make_tree(params)
        # Les fichiers n'existent que le temps de les lire dans le ZIP.
        with TemporaryDirectory(prefix="profils_option_") as directory:
            files = OptionProfiles(tree).plot(directory, show=False)
            return _zip_files(files, params)
    finally:
        del tree
        gc.collect()
