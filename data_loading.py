import os
import GEOparse
import pandas as pd

# 1. Dossier de destination
data_dir = "./data/raw"
os.makedirs(data_dir, exist_ok=True)

# 2. Téléchargement des données GEO
print("Téléchargement du dataset GSE5390...")
gse = GEOparse.get_GEO(geo="GSE5390", destdir=data_dir)

# 3. Extraction des métadonnées
metadata = gse.phenotype_data
print("\n--- Colonnes disponibles dans les métadonnées ---")
print(metadata.columns.tolist())

# 4. Identification automatique de la colonne contenant le statut médical
char_cols = [col for col in metadata.columns if 'characteristics' in col or 'title' in col or 'source' in col]
print(f"\nColonnes descriptives trouvées : {char_cols}")

# Afficher les premières lignes des colonnes pertinentes pour vérifier le contenu
print("\nAperçu des données des sujets :")
print(metadata[char_cols].head())

# 5. Extraction de la matrice d'expression
expression_data = gse.pivot_samples('VALUE')

# 6. Sauvegarde des données brutes
metadata.to_csv(os.path.join(data_dir, "metadata.csv"))
expression_data.to_csv(os.path.join(data_dir, "gene_expression.csv"))
print("\nDonnées sauvegardées avec succès dans le dossier data/raw/")