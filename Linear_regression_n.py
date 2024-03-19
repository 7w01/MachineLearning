import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = np.genfromtxt('data1.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]
x_data = np.concatenate((x_data, np.ones((len(x_data),1))), axis=1)


class LR():
    def __init__(self, lr=0.0000001, epochs=50):
        self.lr = lr
        self.epochs = epochs


    def fit(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
        self.d = len(x_data[0])
        self.m = len(x_data)
        self.theta = np.zeros((len(x_data[0]),), dtype=np.float64)

        for i in range(self.epochs):
            self.gradient_descent()


    def prediction(self, x):
        x = np.concatenate((x, np.ones(len(x), 1)), axis=1)
        X = np.mat(x)
        y = (X * self.theta.T).flatten()
        return y


    def plot(self):
        x = self.x[:, :-1]
        x0, x1 = np.meshgrid(x[:, 0], x[:, 1])

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], self.y, c='r', marker='o', s=100)
        ax.plot_surface(x0, x1, self.theta[0] * x0 + self.theta[1] * x1 + self.theta[2])

        plt.show()


    def cost_function(self):
        theta = np.mat(self.theta)
        X = np.mat(self.x)
        return 1 / (2 * self.m) * np.sum(((X * theta.T).flatten() - self.y) ** 2)


    def gradient_descent(self):
        X = np.mat(self.x)
        for i in range(self.d - 1):
            theta = np.mat(self.theta)
            self.theta[i] -= self.lr * 1 / self.m * np.dot(np.array(X * theta.T).flatten() - self.y, self.x[:, i])
        theta = np.mat(self.theta)
        self.theta[-1] -= self.lr * 1 / self.m * np.sum(np.array(X * theta.T).flatten() - self.y)


l_r = LR()
l_r.fit(x_data, y_data)
l_r.plot()







