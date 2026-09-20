import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Chargement des données
metadata = pd.read_csv("./data/raw/metadata_labeled.csv", index_col=0)
gene_expr = pd.read_csv("./data/raw/gene_expression.csv", index_col=0)

X = gene_expr.T
y = metadata['target']

# 2. Normalisation (Essentiel pour KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Approche 1 : KNN directement sur les gènes ---
# k=3 est un bon choix pour n=15
knn_raw = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_raw.fit(X_scaled, y)
y_pred_raw = knn_raw.predict(X_scaled)

print("--- KNN sur toutes les variables (Brut) ---")
print(f"Accuracy : {accuracy_score(y, y_pred_raw):.2f}")

# --- Approche 2 : KNN sur composantes PCA (Recommandé) ---
pca = PCA(n_components=3) # Réduction à 3 composantes principales
X_pca = pca.fit_transform(X_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_pca.fit(X_pca, y)
y_pred_pca = knn_pca.predict(X_pca)

print("\n--- KNN sur Composantes PCA (Optimisé) ---")
print(f"Accuracy : {accuracy_score(y, y_pred_pca):.2f}\n")
print(classification_report(y, y_pred_pca, target_names=['Control', 'Down Syndrome']))

# 3. Matrice de Confusion KNN (PCA)
cm = confusion_matrix(y, y_pred_pca)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Control', 'Down Syndrome'], 
            yticklabels=['Control', 'Down Syndrome'])
plt.title('Matrice de Confusion - KNN (sur PCA)')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.tight_layout()
plt.savefig("./knn_confusion_matrix.png", dpi=300)
plt.show()