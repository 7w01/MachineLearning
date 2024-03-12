import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# input data
data = np.genfromtxt('data1.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

# learning rate
lr = 0.0000001

# epochs
epochs = 50

# parameters
theta = np.zeros((3,), dtype=np.float64)


# least square
def cost_function(theta, x_data, y_data):
    error = 0
    for i in range(0, len(x_data)):
        error_temp = y_data[i]
        for j in range(0, len(theta)):
            if j == 2:
                error_temp -= theta[j]
            else:
                error_temp -= theta[j] * x_data[i][j]
        error_temp = error_temp ** 2
        error += error_temp
    return error / (2 * float(len(x_data)))


def gradient_descent(theta, x_data, y_data, lr, epochs):
    m = float(len(x_data))
    for e in range(0, epochs):
        theta_temp = np.zeros((3,), dtype=np.float64)
        for i in range(len(x_data)):
            theta_temp[0] += -(1 / m) * x_data[i][0] * (y_data[i] - (theta[0] * x_data[i][0] + theta[1] * x_data[i][1] + theta[2]))
            theta_temp[1] += -(1 / m) * x_data[i][1] * (y_data[i] - (theta[0] * x_data[i][0] + theta[1] * x_data[i][1] + theta[2]))
            theta_temp[2] += -(1 / m) * (y_data[i] - (theta[0] * x_data[i][0] + theta[1] * x_data[i][1] + theta[2]))
        theta = theta - lr * theta_temp
    return theta


# modeling
theta = gradient_descent(theta, x_data, y_data, lr, epochs)

# gragh
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)

x0 = x_data[:, 0]
x1 = x_data[:, 1]

x0, x1 = np.meshgrid(x0, x1)
z = theta[0] * x0 + theta[1] * x1 + theta[2]
ax.plot_surface(x0, x1, z)

plt.show()