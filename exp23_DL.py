import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
np.random.seed(0)
N = 200
theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
r = 2 * theta + np.pi
data1 = np.array([r * np.cos(theta), r * np.sin(theta)]).T

theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
r = -2 * theta - np.pi
data2 = np.array([r * np.cos(theta), r * np.sin(theta)]).T

X = np.vstack([data1, data2])
y = np.hstack([np.zeros(N), np.ones(N)])

model = MLPClassifier(
    hidden_layer_sizes=(3, 3, 3),
    activation='logistic',   # Sigmoid
    learning_rate_init=0.1,
    max_iter=2000
)

model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("Exp 23: Spiral Data (Sigmoid)")
plt.show()

print("Accuracy:", model.score(X, y))
