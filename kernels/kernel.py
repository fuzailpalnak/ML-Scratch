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

    def classifier_product_rule(self, alpha, true_labels):
        """
        Classifier product rule
        w = [sum i=0->n] alpha_{i} * true_labels_{i} * K(x_{i}, x_{t})

        alpha is of shape (n)
        n = number of data points

        true_labels is of shape (n)
        n = number of data points

        :param alpha:
        :param true_labels:
        :return:
        """

        return (alpha * true_labels).dot(self.input_kernel_matrix)

    def compute_kernel_matrix(self):
        raise NotImplementedError


class RBF(Kernel):
    def __init__(self, input_mat, length_scale, output_variance=1):
        """

        Dimension of input_mat = (n X m)
        m = number of features
        n = number of data points

        :param input_mat:
        :param length_scale: determines the length of the 'wiggles' in your function
        :param output_variance:  determines the average distance of your function away from its mean
        """
        super().__init__(input_mat)
        self.length_scale = length_scale
        self.output_variance = output_variance
        self.input_kernel_matrix = self.compute_kernel_matrix()

    def compute_kernel_matrix(self):
        """
        Computation of Training Kernel Matrix, [all data points i->n; K(Xi, Xt)]

        During training Xt is noting but the training points, so this kernel matrix can be computed just once.

        :return:
        """
        n, m = self.input_mat.shape
        g_matrix = self.input_mat.dot(self.input_mat.T)

        g_tile = np.tile(np.diag(g_matrix), (n, 1))
        g_tile = g_tile + g_tile.T - 2 * g_matrix
        rbf_kernel_matrix = self.output_variance * np.exp(-(g_tile / (2 * self.length_scale ** 2)))
        return rbf_kernel_matrix

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
        return self.output_variance * np.exp(-(distance / (2 * self.length_scale ** 2)))


class Poly(Kernel):
    def __init__(self, input_mat, degree):
        """

        Dimension of input_mat = (n X m)
        m = number of features
        n = number of data points

        :param input_mat:
        :param degree:
        """
        super().__init__(input_mat)
        self.degree = degree
        self.input_kernel_matrix = self.compute_kernel_matrix()

    def compute_kernel_matrix(self):
        """
        Computation of Training Kernel Matrix, [all data points i->n; K(Xi, Xt)]

        During training Xt is noting but the training points, so this kernel matrix can be computed just once.

        :return:
        """
        g_matrix = self.input_mat.dot(self.input_mat.T)
        g_matrix = 1 + g_matrix
        poly_kernel = g_matrix ** self.degree
        return poly_kernel

    def decision(self, z, **kwargs):
        """
        This is used for prediction, prediction for just one test point at a time is supported

        z = (1 x m)
        m = number of features

        :param z:
        :param kwargs:
        :return:
        """
        return self.input_mat.dot(z) ** 2
