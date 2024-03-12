import numpy as np


data = np.genfromtxt('data_clustering.csv', delimiter=',')


class KMeans:
    def __init__(self, k):
        self.k = k


    def fit(self, data, max_iteration=10):
        self.data = data
        self.max_iteration = max_iteration
        self.closest_vector = np.zeros((len(data), ))

        # generate random central vectors
        random_data = np.random.permutation(len(self.data))
        self.central_vectors = data[random_data[:self.k], :]

        for e in range(self.max_iteration):
            # compute euclid distance from data to vectors
            dis = self.euclid_dis(self.data)
            for i in range(len(self.data)):
                self.closest_vector[i] = dis[i].argmin()

            # compute new central vectors
            central_vectors_temp = np.array([])
            for i in range(self.k):
                index = self.closest_vector == i
                central_vectors_temp = np.append(central_vectors_temp, np.mean(self.data[index, :], axis=0))

            central_vectors_temp = central_vectors_temp.reshape((self.k, len(data[0])))

            # whether the vectors have changed
            if (central_vectors_temp == self.central_vectors).all():
                break
            else:
                self.central_vectors = central_vectors_temp


    def euclid_dis(self, x):
        dis = np.array([])
        for i in range(len(x)):
            dis = np.append(dis, np.sqrt(np.sum((x[i] - self.central_vectors) ** 2, axis=1)).flatten())
        dis = dis.reshape((len(x), self.k))

        return dis


    def prediction(self, x):
        dis = self.euclid_dis(x)
        return self.central_vectors[dis[0].argmin()]


k_means = KMeans(3)
k_means.fit(data)
cluster = k_means.prediction(np.array([[2.1, 0]]))
print(k_means.central_vectors)
print(cluster)
print(k_means.closest_vector)
