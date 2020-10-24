import collections
import math
from typing import List

Node = collections.namedtuple("Node", ["point", "axis", "label", "left", "right"])
neighbour = collections.namedtuple("Neighbour", ["point", "label", "shortest_distance"])


def square_distance(a, b):
    s = 0
    for x, y in zip(a, b):
        s += (x - y) ** 2
    return s


class KDTree:
    def __init__(self, k: int, points: List):
        """

        :param k:
        :param points: [((x_1, y_1), label_1), ((x_2, y_2), label_2), ......, ((x_n, y_n), label_m)]
        """
        self.root = self.construct_tree(k, (list(points)), axis=0)

    def construct_tree(self, k, points, axis):
        if not points:
            return None

        # Find the point along which the split is going to be performed
        # Find the median along the axis of split
        points.sort(key=lambda o: o[0][axis])
        median_idx = len(points) // 2
        median_point, median_label = points[median_idx]

        # Perform Split Along Axis
        # New split axis can be computed as [A = D mod K]  -> D = depth, K
        # This is also equivalent to [A = (current_axis + 1) mode K]
        next_axis = (axis + 1) % k
        return Node(
            median_point,
            axis,
            median_label,
            self.construct_tree(k, points[:median_idx], next_axis),
            self.construct_tree(k, points[median_idx + 1 :], next_axis),
        )

    def recursive_search(self, root, search_neighbour_for_node, best_neighbour):

        if root is None:
            return best_neighbour

        point, axis, label, left, right = root

        # Compute the square distance between current point and point for which neighbour is to be found
        # To store if nearest neighbour has already been visited
        here_sd = square_distance(point, search_neighbour_for_node)

        # if new node with shortest square dist is found update the neighbour list
        if here_sd < best_neighbour.shortest_distance:
            best_neighbour = neighbour(point, label, here_sd)

        # Check the distance between the search node and current point node
        # to decide which side of the tree to traverse to find nearest neighbour
        diff = search_neighbour_for_node[axis] - point[axis]

        # if diff is less than zero traverse in left side of the tree else right
        immediate_search_space, fall_back_search_space = (
            (left, right) if diff <= 0 else (right, left)
        )
        best_neighbour = self.recursive_search(
            immediate_search_space, search_neighbour_for_node, best_neighbour
        )

        # if the distance to root is smaller than the distance to nearest neighbour then the algorithm has to look
        # on the remaining half of the tree i.e the `fall_back_search_space`
        if diff ** 2 < best_neighbour.shortest_distance:
            best_neighbour = self.recursive_search(
                fall_back_search_space, search_neighbour_for_node, best_neighbour
            )

        return best_neighbour

    def nearest_neighbor(self, search_neighbour_for_node):

        # state of search: best point found, its label,
        # lowest squared distance
        my_neighbour = self.recursive_search(
            self.root, search_neighbour_for_node, neighbour(None, None, float("inf"))
        )
        return (
            my_neighbour.point,
            my_neighbour.label,
            math.sqrt(my_neighbour.shortest_distance),
        )
