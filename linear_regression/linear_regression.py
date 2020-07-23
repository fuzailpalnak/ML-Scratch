import numpy as np
import matplotlib.pyplot as plt

# https://www.onlinegdb.com/online_python_compiler


class LinearRegression:
    def __init__(self):
        self.lr = 0.001
        self.num_iter = 10000
        self.w = np.array([0.0, 0.0])
        self.x = list()
        self.y = list()
        self.data = None
        self.populate_data()

    @staticmethod
    def __decision_rule(w, x):
        return np.dot(w, x)

    def __gradient(self, w, x, y):
        return 2 * np.multiply((self.__decision_rule(w, x) - y), x)

    def __loss(self, w, x, y):
        return np.square(self.__decision_rule(w, x) - y)

    def train(self):
        for num_iter in range(self.num_iter):
            for iterator in range(len(self.data)):
                x = self.x[iterator]
                y = self.y[iterator]

                gradient = self.__gradient(self.w, x, y)
                self.w -= self.lr * gradient
                print("Step: {} Loss : {}".format(num_iter, self.__loss(self.w, x, y)))
            # self.plt_decision_boundary()

    def slope_intercept(self):
        a, b = tuple(self.w)
        return a, b

    def populate_data(self):
        self.data = self.get_data(10)

        for i in range(len(self.data)):
            data = [self.data[i][0], 1]
            self.y.append(np.array(self.data[i][1]))
            self.x.append(data)

    def plt_decision_boundary(self):
        slope, intercept = self.slope_intercept()

        x = np.linspace(0, 1)
        y = slope * x + intercept
        plt.plot(x, y, "-r", label="decision boundary")
        plt.scatter(self.data[:, 0], self.data[:, 1], marker="o")
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
        data = self.plt_data_t("DATA", number_of_points)
        plt.show()
        return data


p = LinearRegression()
p.train()
p.plt_decision_boundary()
