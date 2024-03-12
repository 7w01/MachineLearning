import numpy as np

ws = np.array([[1,2,3],[4,5,6],[7,8,9]])

ws = np.mean(ws, axis=1)
print(ws)