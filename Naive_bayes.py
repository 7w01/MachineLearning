import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.DataFrame(pd.read_csv('data3.csv'))


class NaiveBayes():
    #def __init__(self):


    def fit(self, data):
        self.data = data
        self.m = len(data)
        self.d = len(self.data.columns) - 1
        self.ds = self.data.columns
        self.d_vals = {}

        self.P_c = self.prior_pro()

        for d_i in self.data.columns[:-1]:
            temp_frame = self.con_pro(d_i)
            temp_frame['iden'] = d_i
            if d_i == self.data.columns[0]:
                self.P_xi_c = temp_frame
            else:
                self.P_xi_c = pd.concat([self.P_xi_c, temp_frame])

        self.P_xi_c = self.P_xi_c.reset_index()
        self.P_xi_c.rename(columns={'index':'x_value'}, inplace=True)


    def prior_pro(self):
        y = self.data['y']
        self.C_vals = y.unique()
        P_c = {}
        for c_k in self.C_vals:
            P_c[c_k] = sum(y == c_k) / len(y)

        return P_c


    def con_pro(self, identity):
        di_vals = self.data[identity].unique()
        self.d_vals[identity] = di_vals
        frame = self.data[[identity, 'y']]
        P_xi_c = {}

        for c_k in self.C_vals:
            temp_c_k = {}
            for di_val in di_vals:
                num1 = len(frame[frame.iloc[:, 1] == c_k])
                num2 = len(frame[(frame.iloc[:, 1] == c_k) & (frame.iloc[:, 0] == di_val)])
                temp_c_k[di_val] = num2 / num1
            P_xi_c[c_k] = temp_c_k

        return pd.DataFrame(P_xi_c)


    def prediction(self, x):
        C_pro = {}
        for c_k in self.C_vals:
            c_k_pro = self.prior_pro()[c_k]
            for i in range(self.d):
                c_k_pro *= self.P_xi_c.loc[(self.P_xi_c['x_value'] == x[i]) & (self.P_xi_c['iden'] == self.ds[i]), c_k].values[0]
            C_pro[c_k] = c_k_pro

        for key, value in C_pro.items():
            if value == max(C_pro.values()):
                max_c = key
                break

        return max_c, C_pro


NB = NaiveBayes()
NB.fit(data)
print(NB.P_xi_c)

x_sample = [2, 'S']
print(NB.prediction(x_sample))

