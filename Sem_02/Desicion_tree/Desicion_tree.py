import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

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

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Datos Sintéticos para Árbol de Decisión")
plt.show()

# Crear el modelo de árbol de decisión
decision_tree = DecisionTreeClassifier()

# Entrenar el modelo con los datos sintéticos
decision_tree.fit(X, y)

# Definir los límites del gráfico
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Crear una grilla de valores
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predecir cada punto en la grilla
Z = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap="Paired", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Frontera de Decisión del Árbol de Decisión")
plt.show()
