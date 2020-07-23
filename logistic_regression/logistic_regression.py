import numpy as np
import matplotlib.pyplot as plt

# https://www.onlinegdb.com/online_python_compiler


class LogisticRegression:
    def __init__(self):
        self.lr = 0.01
        self.num_iter = 10
        self.w = np.array([0.0, 0.0, 0.0])
        self.data = list()
        self.labels = list()
        self.positive = None
        self.negative = None
        self.populate_data()

    @staticmethod
    def __decision_rule(w, x):
        return np.dot(w, x)

    def __sigmoid(self, w, x, y):
        z = self.__decision_rule(w, x)
        return 1 / (1 + np.exp(-z * y))

    def __expo(self, w, x, y):
        z = self.__decision_rule(w, x)
        return np.exp(-z * y)

    def __gradient(self, w, x, y):
        return self.__sigmoid(w, x, y) * self.__expo(w, x, y) * (-y * x)

    def __loss(self, w, x, y):
        return np.log(1 + self.__expo(w, x, y)).mean()

    def predict_prob(self, x):
        return self.__sigmoid(self.w, x, 1)

    def predict(self, x, threshold):
        return self.predict_prob(x) >= threshold

    def train(self):
        for num_iter in range(self.num_iter):
            for iterator in range(len(self.data)):
                x = self.data[iterator]
                y = self.labels[iterator]

                gradient = self.__gradient(self.w, x, y)
                self.w -= self.lr * gradient
                print("Step: {} Loss : {}".format(num_iter, self.__loss(self.w, x, y)))
            self.plt_decision_boundary()

    def slope_intercept(self):
        a, b, c = tuple(self.w)
        return -a / b, -c / b

    def populate_data(self):
        self.positive, self.negative = self.get_data(10)

        for i in range(len(self.positive)):
            data = [self.positive[i][0], self.positive[i][1], 1]
            self.data.append(np.array(data))
            self.labels.append(1)
        for i in range(len(self.negative)):
            data = [self.negative[i][0], self.negative[i][1], 1]
            self.data.append(np.array(data))
            self.labels.append(-1)

    def plt_decision_boundary(self):
        slope, intercept = self.slope_intercept()

        x = np.linspace(0, 1)
        y = slope * x + intercept
        plt.plot(x, y, "-r", label="decision boundary")
        plt.scatter(self.positive[:, 0], self.positive[:, 1], marker="o")
        plt.scatter(self.negative[:, 0], self.negative[:, 1], marker="x")
        plt.title("Graph of Decision Boundary")
        plt.xlabel("x", color="#1C2833")
        plt.ylabel("y", color="#1C2833")
        plt.legend(loc="upper left")
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plt_data_t(title, number_of_points):
        plt.title(title, fontsize=10)
        plt.draw()
        pts = np.asarray(plt.ginput(number_of_points, timeout=-1))
        return pts

    def get_data(self, number_of_points):
        plt.clf()
        plt.setp(plt.gca(), autoscale_on=False)
        positive = self.plt_data_t("Positive Class", number_of_points)
        negative = self.plt_data_t("Negative Class", number_of_points)

        plt.title("DATA", fontsize=10)
        plt.scatter(positive[:, 0], positive[:, 1], marker="o")
        plt.scatter(negative[:, 0], negative[:, 1], marker="x")

        plt.draw()
        plt.show()
        return positive, negative


p = LogisticRegression()
p.train()
p.plt_decision_boundary()
