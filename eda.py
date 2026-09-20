import pandas as pd
import numpy as np

# 1. Charger les métadonnées et la matrice d'expression
metadata = pd.read_csv("./data/raw/metadata.csv", index_col=0)
gene_expr = pd.read_csv("./data/raw/gene_expression.csv", index_col=0)

# 2. Créer la variable cible (Target) à partir de la colonne 'title'
# 1 = Down Syndrome, 0 = Control
metadata['target'] = metadata['title'].apply(
    lambda x: 1 if 'down_syndrome' in str(x).lower() else 0
)

# 3. Afficher la répartition des classes
print("--- Répartition des classes ---")
print(metadata['target'].value_counts().rename({1: 'Down Syndrome', 0: 'Control'}))

# 4. Vérification des dimensions des matrices
print("\n--- Dimensions des données ---")
print(f"Nombre de sujets (métadonnées) : {metadata.shape[0]}")
print(f"Nombre de gènes analysés      : {gene_expr.shape[0]}")
print(f"Nombre d'échantillons en tout : {gene_expr.shape[1]}")

# 5. Sauvegarder les métadonnées annotées
metadata.to_csv("./data/raw/metadata_labeled.csv")
print("\nMétadonnées annotées enregistrées sous metadata_labeled.csv !")