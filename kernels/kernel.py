import numpy as np
from scipy.spatial.distance import pdist, squareform


class Kernel:
    def __init__(self, input_mat):
        """

        Dimension of input_mat = (n X m)
        m = number of features
        n = number of data points


        input_kernel_matrix is computation of Training Kernel Matrix, [all training data points i->n; K(Xi, Xt)],
        The values in the first row of input_kernel_matrix corresponds to row_1 (n X 1) = [c1, c2, c3, ...., cn],
        where the columns are kernel operation of each training data point with Xt.

        input_kernel_matrix = [[0, c2, c3, ...., cn = [all training data points i->n; K(Xi, X1)]
                               c2  0                = [all training data points i->n; K(Xi, X2)]
                               c3,    0
                               c3,
                               ..          ...
                               cn                0]]

        During training Xt is noting but the training points, so this kernel matrix can be computed just once.

        :param input_mat:
        """
        self.input_mat = input_mat
        self.input_kernel_matrix = None

    def decision(self, z, **kwargs):
        """
        This is used for prediction, prediction for just one test point at a time is supported

        z = (1 X m)
        m = number of features

        :param z:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def product_rule(self, alpha, true_labels):
        """
        alpha is of shape (n X 1)
        n = number of data points

        true_labels is of shape (n X 1)
        n = number of data points

        :param alpha:
        :param true_labels:
        :return:
        """
        return np.sum(alpha * self.input_kernel_matrix * true_labels.T, axis=1)

    def compute_training_kernel_matrix(self):
        raise NotImplementedError


class RBF(Kernel):
    def __init__(self, input_mat, sigma):
        """

        Dimension of input_mat = (n X m)
        m = number of features
        n = number of data points

        :param input_mat:
        :param sigma:
        """
        super().__init__(input_mat)
        self.sigma = sigma
        self.input_kernel_matrix = self.compute_training_kernel_matrix()

    def compute_training_kernel_matrix(self):
        """
        Computation of Training Kernel Matrix, [all data points i->n; K(Xi, Xt)]

        During training Xt is noting but the training points, so this kernel matrix can be computed just once.

        :return:
        """
        distance = pdist(self.input_mat, "sqeuclidean",)
        sq_form = squareform(np.exp((distance / (2 * self.sigma ** 2))))
        np.fill_diagonal(sq_form, sq_form.diagonal() + 1)
        return sq_form

    def decision(self, z, **kwargs):
        """
        This is used for prediction, prediction for just one test point at a time is supported

        z = (1 x m)
        m = number of features

        :param z:
        :param kwargs:
        :return:
        """
        distance = np.linalg.norm(self.input_mat - z, axis=1, ord=2) ** 2
        return np.exp((distance / (2 * self.sigma ** 2)))
