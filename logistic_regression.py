import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

# input
data = np.genfromtxt('data2.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1, np.newaxis]


def plot():
    p0 = []
    p1 = []
    for i in range(len(y_data)):
        if y_data[i][0] == 1:
            p0.append(x_data[i])
        else:
            p1.append(x_data[i])

    p0 = np.array(p0)
    p1 = np.array(p1)

    # gragh
    ax = plt.figure().add_subplot(111, projection='3d')
    scatter0 = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c='r', marker='x')
    scatter1 = ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='b', marker='o')

    x0_test = [0, 1, 2, 3]
    x1_test = [0, 1, 2, 3]
    x0_test, x1_test = np.meshgrid(x0_test, x1_test)
    x2_test = -(Ws[0] * x0_test + Ws[1] * x1_test + Ws[3]) / Ws[2]
    ax.plot_surface(x0_test, x1_test, x2_test)

    plt.legend(handles=[scatter1, scatter0], labels=['label1', 'label0'], loc='best')


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cost_function(x_mat, y_mat, Ws):
    return -1 / float(len(y_data)) * np.sum(np.multiply(y_mat, x_mat * Ws) - np.log(1 + np.exp(1 + x_mat * Ws)))


def gradient_decsent(x_data, y_data, lr, epochs, Ws):
    x_mat = np.mat(x_data)
    y_mat = np.mat(y_data)
    Ws = np.mat(Ws)

    for i in range(epochs):
        Ws = Ws - lr * x_mat.T * (sigmoid(x_mat * Ws) - y_mat) / float(len(y_data))

    Ws = np.array(Ws)
    Ws = Ws[:, 0]

    return Ws


# add intercept
x_data = np.concatenate((x_data, np.ones((len(y_data), 1))), axis=1)

# learning rate
lr = 0.1

# epochs
epochs = 10000

# parameters
Ws = np.ones((len(x_data[0]), 1))

Ws = gradient_decsent(x_data, y_data, lr, epochs, Ws)

plot()

print(Ws)

plt.show()
