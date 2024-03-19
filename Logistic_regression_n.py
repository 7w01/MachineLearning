import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = np.genfromtxt('data2.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]
x_data = np.concatenate((x_data, np.ones((len(x_data), 1))), axis=1)


class LogisticRegression():
    def __init__(self, lr=0.1, epochs=10000):
        self.lr = lr
        self.epochs = epochs


    def fit(self, x, y):
        self.x = x
        self.y = y
        self.m = len(x)
        self.d = len(x[0])
        self.theta = np.zeros((len(x[0]), ))

        for i in range(self.epochs):
            self.gradient_descent()


    #def cost_function(self):


    def prediction(self, x):
        x = np.concatenate((x,np.ones((len(x), 1))), axis=1)
        X = np.mat(x)
        theta = np.mat(self.theta)
        y = self.sigmoid(np.array(X * theta.T).flatten())
        return y


    def plot(self):
        p0 = []
        p1 = []
        for i in range(self.m):
            if self.y[i] == 1:
                p1.append(self.x[i])
            else:
                p0.append(self.x[i])

        p0 = np.array(p0)
        p1 = np.array(p1)

        ax = plt.figure().add_subplot(111, projection='3d')
        scatter1 = ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='b', marker='o', s=50)
        scatter2 = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c='r', marker='o', s=50)

        x = self.x[:, :-1]
        x0, x1 = np.meshgrid(x[:, 0], x[:, 1])

        ax.plot_surface(x0, x1, -1 / self.theta[2] * (self.theta[0] * x0 + self.theta[1] * x1 + self.theta[-1]))

        plt.show()


    def gradient_descent(self):
        X = np.mat(self.x)
        for i in range(self.d - 1):
            theta = np.mat(self.theta)
            self.theta[i] -= self.lr / self.m * np.dot(self.sigmoid(np.array(X * theta.T).flatten()) - self.y, self.x[:, i])
        self.theta[-1] -= self.lr / self.m * np.sum(self.sigmoid(np.array(X * theta.T).flatten()) - self.y)


    @staticmethod
    def sigmoid(x):
       return 1 / (1 + np.exp(-x))


L_R = LogisticRegression()
L_R.fit(x_data, y_data)
L_R.plot()







