import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

# generate sample data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

# plot the data
X = np.array(X)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", marker="o", edgecolor="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Blobs of Data")
plt.show()


# Crear el modelo de Árbol de Decisión
decision_tree = DecisionTreeClassifier(max_depth=5)  # max_depth controla la complejidad

# Entrenar el modelo con los datos sintéticos
decision_tree.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", marker="o", edgecolor="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Blobs of Data")
plt.show()
