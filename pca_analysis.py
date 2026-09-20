import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Chargement des données
metadata = pd.read_csv("./data/raw/metadata_labeled.csv", index_col=0)
gene_expr = pd.read_csv("./data/raw/gene_expression.csv", index_col=0)

# Alignement des colonnes d'expression avec l'ordre des métadonnées
X = gene_expr.T  # Transposition : lignes = échantillons (15), colonnes = gènes (22283)
y = metadata['target']

# 2. Normalisation des données (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Application de la PCA (2 composantes)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Création du DataFrame pour le graphique
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Condition'] = y.map({0: 'Control', 1: 'Down Syndrome'}).values

# 5. Visualisation
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Condition', 
    style='Condition',
    data=df_pca, 
    s=150, 
    palette={'Control': 'blue', 'Down Syndrome': 'red'}
)

var_exp = pca.explained_variance_ratio_ * 100
plt.title(f'PCA - Expression Génique (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)')
plt.xlabel(f'Composante Principale 1 ({var_exp[0]:.1f}% variance)')
plt.ylabel(f'Composante Principale 2 ({var_exp[1]:.1f}% variance)')
plt.grid(True, linestyle='--', alpha=0.5)

# Sauvegarde de la figure pour le portfolio
plt.savefig("./pca_visualization.png", dpi=300, bbox_inches='tight')
print("Graphique PCA sauvegardé sous 'pca_visualization.png' !")
plt.show()

# 6. Extraction des 10 gènes influençant le plus PC1 (loadings)
loadings = pd.Series(np.abs(pca.components_[0]), index=gene_expr.index)
top_genes = loadings.sort_values(ascending=False).head(10)

print("\n--- Top 10 des gènes les plus discriminants (PC1) ---")
print(top_genes)