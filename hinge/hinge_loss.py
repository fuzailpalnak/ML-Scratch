import numpy as np
import matplotlib.pyplot as plt

# https://www.onlinegdb.com/online_python_compiler


class Hinge:
    def __init__(self):
        self.lr = 0.0001
        self.num_iter = 500000
        self.w = np.array([0.0, 0.0])
        self.b = np.array([1.0])
        self.data = list()
        self.labels = list()
        self.positive = None
        self.negative = None
        self.populate_data()

    @staticmethod
    def __decision_rule(w, b, x):
        return np.dot(w, x) + b

    def __gradient(self, w, b, x, y, weight_decay):
        if self.__decision_rule(w, b, x) * y < 1:
            w_grad = -2 * x * y * (1 - y * self.__decision_rule(w, b, x)) + (
                2 * weight_decay * w
            )
            b_grad = -2 * y * (1 - y * self.__decision_rule(w, b, x))
        else:
            w_grad = np.array([0.0, 0.0])
            b_grad = np.array([0.0])
        return w_grad, b_grad

    def __loss(self, w, b, x, y):
        return np.maximum(0, 1 - ((self.__decision_rule(w, b, x)) * y)) ** 2

    def predict_prob(self, x):
        raise NotImplementedError

    def predict(self, x, threshold):
        raise NotImplementedError

    def train(self, weight_decay=0.01):
        for num_iter in range(self.num_iter):
            loss = list()
            for iterator in range(len(self.data)):
                x = self.data[iterator]
                y = self.labels[iterator]

                w_grad, b_grad = self.__gradient(self.w, self.b, x, y, weight_decay)
                self.w -= self.lr * w_grad
                self.b -= self.lr * b_grad
                loss.append(self.__loss(self.w, self.b, x, y))
            print(
                "Step: {} Loss : {} with w: {}, b: {}".format(
                    num_iter, np.array(loss).mean(), self.w, self.b
                )
            )
            # self.plt_decision_boundary()

    def slope_intercept(self):
        a, b = tuple(self.w)
        c = self.b
        return -(a / b), -(c / b)

    def populate_data(self):
        self.positive, self.negative = self.get_data(10)

        for i in range(len(self.positive)):
            data = [self.positive[i][0], self.positive[i][1]]
            self.data.append(np.array(data))
            self.labels.append(1)
        for i in range(len(self.negative)):
            data = [self.negative[i][0], self.negative[i][1]]
            self.data.append(np.array(data))
            self.labels.append(-1)

    def plt_decision_boundary(self):
        slope, intercept = self.slope_intercept()
        x = np.linspace(0, 10)
        y = slope * x + intercept

        plt.plot(x, y, "-r", label="decision boundary")

        plt.axis([0, 10, 0, 20])
        plt.scatter(self.positive[:, 0], self.positive[:, 1], marker="o")
        plt.scatter(self.negative[:, 0], self.negative[:, 1], marker="x")
        plt.title("Graph of Decision Boundary")
        plt.xlabel("x", color="#1C2833")
        plt.ylabel("y", color="#1C2833")
        plt.legend(loc="upper left")
        plt.axis("equal")
        plt.show()

    @staticmethod
    def plt_data_t(title, number_of_points):
        plt.title(title, fontsize=10)
        plt.draw()
        pts = np.asarray(plt.ginput(number_of_points, timeout=-1))
        return pts

    def get_data(self, number_of_points):
        plt.clf()
        plt.axis([0, 10, 0, 20])
        plt.setp(plt.gca(), autoscale_on=False)
        positive = self.plt_data_t("Positive Class", number_of_points)
        negative = self.plt_data_t("Negative Class", number_of_points)

        plt.title("DATA", fontsize=10)
        plt.scatter(positive[:, 0], positive[:, 1], marker="o")
        plt.scatter(negative[:, 0], negative[:, 1], marker="x")

        plt.draw()
        plt.show()
        return positive, negative


p = Hinge()
p.train()
p.plt_decision_boundary()
