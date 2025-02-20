import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42,
)

X = np.array(X)  # asegurarse de que X sea un array de numpy

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired", edgecolors="k")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Datos Sintéticos para Árbol de Decisión")
plt.show()
