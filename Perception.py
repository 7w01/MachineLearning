import numpy as np


class Multilayer_Perception:
    def _init_(self, data, label, layer, normalize_data = False, max_iterations = 1000, alpha = 0.1):
       self.data = data
       self.label = label
       self.layer = layer
       self.theta = Multilayer_Perception.theta_init(layer)
       self.max_iterations = 1000
       self.alpha = 0.1


    @staticmethod
    def theta_init(self, layer):
        layer_num = len(layer)
        theta = {}

        for i in range(layer_num - 1):
            in_count = layer[i]
            out_count = layer[i + 1]
            # 考虑偏置项
            theta[i] = np.random.rand(out_count, in_count + 1) * 0.05

        return theta


    def train(self):
        return



    def gradient_descent(self, unrolled_theta):
        optimized_theta = unrolled_theta
        cost_history = []
        for i in range(self.max_iterations):
            cost = self.cost_function()
            cost_history.append(cost)





    def cost_function(self):
        layers_num = len(self.layer)
        example_num = len(self.data)
        label_num = self.layer[-1]

        prediction = self.forward_propagation()
        bitwise_label = np.zeros((example_num, label_num))
        for i in range(len(self.data)):
            bitwise_label[i][self.label[i][0]] = 1

        return 1.0 / example_num * (np.sum(-np.log(prediction[bitwise_label == 1])) + np.sum(-np.log(1 - prediction[bitwise_label == 0])))


    def unroll_theta(self):
        unrolled_theta = np.array([])
        for i in range(len(self.theta)):
            theta_temp = np.array(self.theta[i])
            unrolled_theta = np.append(unrolled_theta, theta_temp.flatten(), axis=0)

        return unrolled_theta


    def roll_theta(self, unrolled_theta):
        layer_num = len(self.layer)
        theta = {}
        index = 0

        for i in range(layer_num - 1):
            theta_width = self.layer[i]
            theta_height = self.layer[i + 1]
            theta_num = theta_width * theta_height

            start = index
            end = index + theta_num

            theta[i] = unrolled_theta[start:end].reshape((theta_height, theta_width))
            index += theta_num

            self.theta = theta


    def forward_propagation(self):
        layer_num = len(self.layer)
        in_layer = self.data
        for i in range(layer_num - 1):
            theta = self.theta[i]
            out_layer = self.sigmoid(np.dot(in_layer, theta.T))
            out_layer = np.concatenate((out_layer, np.ones((len(self.data), 1))), axis=0)
            in_layer = out_layer

        return in_layer[:, :-1]


    def back_propagation(self):
        layers_num = len(self.layer)
        example_num = len(self.data)
        label_num = self.layer[-1]

        delta = {}

        # initialize delta
        for i in range(layers_num - 1):
            in_count = self.layer[i]
            out_count = self.layer[i + 1]
            delta[i] = np.zeros((out_count, in_count + 1))

        for i in range(example_num):


            for j in range(layers_num - 1):
                layer_theta = self.theta[j]



    def gradient_step(self):
        return



    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))