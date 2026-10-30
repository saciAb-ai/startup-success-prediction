"""Construction de la cible pour la classification des issues des startups.

Projet : Prédiction du succès ou de l'échec des startups à partir de leurs caractéristiques initiales.

Classe 1 : IPO ou acquisition ("ipo", "acquired").
Classe 0 : fermeture ("closed").
Statut "operating" : exclu, car l'issue n'est pas connue selon cette definition.

IMPORTANT : la colonne ``status`` sert uniquement a creer ``target``.
Elle ne doit jamais etre utilisee comme variable d'entree des modeles.
"""

import pandas as pd


TARGET_MAPPING = {
    "ipo": 1,
    "acquired": 1,
    "closed": 0,
}
KNOWN_STATUSES = set(TARGET_MAPPING) | {"operating"}


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les startups IPO, acquired et closed avec leur cible binaire.

    Parametres
    ----------
    df : pandas.DataFrame
        Donnees Crunchbase contenant obligatoirement la colonne ``status``.

    Retour
    ------
    pandas.DataFrame
        Copie filtree du DataFrame, avec ``status`` normalise et ``target``.
        Le DataFrame d'origine n'est pas modifie.

    Notes
    -----
    - ``operating`` est exclu, et non classe comme succes ou echec.
    - Les statuts inconnus ou manquants provoquent une erreur explicite.
    - ``status`` et ``target`` ne sont pas des variables d'entree (X).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'argument df doit etre un pandas.DataFrame.")
    if "status" not in df.columns:
        raise ValueError("La colonne obligatoire 'status' est absente.")
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    normalized = df["status"].astype("string").str.strip().str.lower()
    unknown = normalized.loc[~normalized.isin(KNOWN_STATUSES)]
    if not unknown.empty:
        values = sorted(unknown.fillna("<manquant>").unique().tolist())
        raise ValueError(f"Statuts inconnus ou manquants : {values}")

    selected = normalized.isin(TARGET_MAPPING)
    result = df.loc[selected].copy()
    if result.empty:
        raise ValueError("Aucune startup avec le statut IPO, acquired ou closed.")

    result["status"] = normalized.loc[selected]
    result["target"] = result["status"].map(TARGET_MAPPING).astype("int8")
    return result


if __name__ == "__main__":
    # Test simple : aucune donnee Kaggle requise pour verifier cette etape.
    example = pd.DataFrame({
        "status": [" IPO ", "Acquired", "Closed", "Operating"],
        "country_code": ["US", "FR", "DZ", "US"],
    })
    result = prepare_target(example)
    assert len(result) == 3
    assert result["target"].tolist() == [1, 1, 0]
    assert "operating" not in result["status"].tolist()
    assert example.loc[0, "status"] == " IPO "  # L'original reste intact.
    print("TEST OK : IPO=1, Acquired=1, Closed=0, Operating exclu.")
    print("Effectifs du test :", result["target"].value_counts().sort_index().to_dict())
