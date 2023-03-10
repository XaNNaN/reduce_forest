class Tree:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.__check_matrix()

    def __str__(self):
        return f'{self.adjacency_matrix}'

    def __repr__(self):
        return f'{self.adjacency_matrix}'

    def __check_matrix(self):
        if self.adjacency_matrix.ndim != 2:
            raise NotTwoDimensionsInAdjacencyMatrix(self.adjacency_matrix.ndim)

        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise DifferentDimensionsOfAdjacencyMatrixError(self.adjacency_matrix.shape)


# Classes for mistakes in creating adjacency matrix
class DifferentDimensionsOfAdjacencyMatrixError(ValueError):
    pass


class NotTwoDimensionsInAdjacencyMatrix(ValueError):
    pass
