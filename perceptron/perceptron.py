import numpy as np
import matplotlib.pyplot as plt

# https://www.onlinegdb.com/online_python_compiler


class Perceptron:
    def __init__(self):
        self.w = np.array([0.0, 0.0, 0.0])
        self.data = list()
        self.labels = list()
        self.positive = None
        self.negative = None
        self.populate_data()

    @staticmethod
    def __decision_rule(w, x):
        """
        data points above the hyperplane will be positive as the theta will be [0, 90] with respect to self.w
        and points below the hyperplane will be negative

        :param w:
        :param x:
        :return:
        """
        return np.dot(w, x)

    @staticmethod
    def __update(w, x, y):
        """
        if Point belong to -1 class then w = w - x
        if point belong to +1 class then w = w + x
        The main objective here is to increase the cos(alpha) between the weight vector and the positive data points
        and decrease the cos(alpha) for negative points

        :param w:
        :param x:
        :param y:
        :return:
        """
        w += y * x
        return w

    def train(self):
        step = 0
        while True:
            miss_classified = 0
            for iterator in range(len(self.data)):
                x = self.data[iterator]
                y = self.labels[iterator]

                if self.__decision_rule(self.w, x) * y <= 0:
                    # Miss classified the data point and adjust the weight
                    w_prev = self.w
                    self.w = self.__update(self.w, x, y)
                    miss_classified = miss_classified + 1
                    print(
                        "Adjusting Weight from w: {} to w_new: {}".format(
                            tuple(w_prev), tuple(self.w)
                        )
                    )
            # self.plt_decision_boundary()
            step += 1
            if miss_classified == 0:
                # if no miss classified then the perceptron has converged and found a hyperplane
                print("Perceptron Converged on Step : {}".format(step))
                break

    def predict(self, x):
        if self.__decision_rule(self.w, x) >= 0:
            return 1
        else:
            return -1

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

    def slope_intercept(self):
        a, b, c = tuple(self.w)
        return -a / b, -c / b

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


p = Perceptron()
p.train()
print(p.predict(np.array([1, 1, 1])))
p.plt_decision_boundary()
