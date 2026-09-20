import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Chargement des données
metadata = pd.read_csv("./data/raw/metadata_labeled.csv", index_col=0)
gene_expr = pd.read_csv("./data/raw/gene_expression.csv", index_col=0)

X = gene_expr.T  # (15 échantillons x 22283 gènes)
y = metadata['target']

# 2. Entraînement du modèle Random Forest
# n_estimators=100 arbres, random_state pour la reproductibilité
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 3. Prédictions & Évaluation
y_pred = rf_model.predict(X)

print("--- Performances du Random Forest ---")
print(f"Accuracy : {accuracy_score(y, y_pred):.2f}\n")
print("Rapport de classification :")
print(classification_report(y, y_pred, target_names=['Control', 'Down Syndrome']))

# 4. Matrice de Confusion
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Control', 'Down Syndrome'], 
            yticklabels=['Control', 'Down Syndrome'])
plt.title('Matrice de Confusion - Random Forest')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.tight_layout()
plt.savefig("./rf_confusion_matrix.png", dpi=300)
plt.show()

# 5. Top 15 des gènes les plus importants selon Random Forest
importances = rf_model.feature_importances_
feature_series = pd.Series(importances, index=X.columns)
top_genes = feature_series.nlargest(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genes.values, y=top_genes.index, palette='viridis')
plt.title('Top 15 des gènes les plus discriminants (Feature Importance - RF)')
plt.xlabel('Importance relative (Gini Importance)')
plt.ylabel('Sonde / Gène')
plt.tight_layout()
plt.savefig("./rf_feature_importance.png", dpi=300)
plt.show()

print("\n--- Top 5 des gènes retenus par Random Forest ---")
print(top_genes.head(5))