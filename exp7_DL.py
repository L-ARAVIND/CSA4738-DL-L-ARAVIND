import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

plt.plot(x, y, color='pink')

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Visualization of the Sigmoid Function")

plt.show()
