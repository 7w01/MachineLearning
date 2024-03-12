import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data1.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 2]

#plt.scatter(x_data, y_data)

# learning rate
lr = 0.0000001
# intercept
b = 0
# slope
k = 0
# epochs
epochs = 50


# least square
def cost_function(x_data, y_data, k, b):
    m = float(len(x_data))
    error = 0
    for i in range(0, len(x_data)):
        error += (y_data[i] - k * x_data[i] - b) ** 2
    return error / (2 * m)


def gradient_descent(x_data, y_data, lr, k, b, epochs):
    m = float(len(x_data))
    for i in range(0, epochs):
        k_temp = 0
        b_temp = 0
        for j in range(0, len(x_data)):
            k_temp += -(1 / m) * (y_data[j] - k * x_data[j] - b) * x_data[j]
            b_temp += -(1 / m) * (y_data[j] - k * x_data[j] - b)
        k = k - lr * k_temp
        b = b - lr * b_temp
    return k, b


k, b =gradient_descent(x_data, y_data, lr, k, b, epochs)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r')
plt.show()
