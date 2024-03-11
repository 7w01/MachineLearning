import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

data_raw = datasets.load_iris()
data = data_raw['data']
label = data_raw['target']

data_train, data_test, label_train, label_test = train_test_split(data, label)

class KNN:
    def __init__(self, k):
        self.k = k


    def fit(self, data, label):
        self.data = data
        self.label = label


    def predict(self, X):
        result = []

        for x in X:
            dis = np.sqrt(np.sum((x - self.data) ** 2, axis=1))
            # 返回排序索引
            index = dis.argsort()
            index = index[:self.k]
            # 返回数组中每个元素出现的次数
            count = np.bincount(self.label[index])
            # 返回值最大的索引,此处即label
            result.append(count.argmax())

        return np.array(result)


knn = KNN(3)
knn.fit(data_train, label_train)
result = knn.predict(data_test)
print(result)
print(label_test)