import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

desicion_tree = DecisionTreeClassifier(max_depth=3, random_state=42)

desicion_tree.fit(x_train, y_train)

acuaracy = desicion_tree.score(x_test, y_test)


plt.figure(figsize=(12, 8))
plot_tree(
    desicion_tree,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)
plt.show()

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = desicion_tree.predict(new_flower)
print(f"Prediction: {iris.target_names[prediction][0]}")
