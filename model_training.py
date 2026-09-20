import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. Chargement des données
metadata = pd.read_csv("./data/raw/metadata_labeled.csv", index_col=0)
gene_expr = pd.read_csv("./data/raw/gene_expression.csv", index_col=0)

X = gene_expr.T
y = metadata['target']

# 2. Normalisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 3. Entraînement d'un modèle Lasso (Pénalisation L1)
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
print("--- Performance du modèle ---")
print(f"Accuracy : {accuracy_score(y, y_pred):.2f}")
print("\nRapport de classification :")
print(classification_report(y, y_pred, target_names=['Control', 'Down Syndrome']))

# 4. Explicabilité avec SHAP (Feature Importance)
explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Sauvegarde du graphique SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_scaled, max_display=15, show=False)
plt.title("Top 15 des gènes impactant la prédiction de la Trisomie 21", fontsize=12)
plt.tight_layout()
plt.savefig("./shap_summary.png", dpi=300)
print("\nGraphique SHAP sauvegardé sous 'shap_summary.png' !")
plt.show()