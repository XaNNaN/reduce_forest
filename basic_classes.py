import numpy as np
import random
import graphviz


def plot_matrix(matrix, fname="plot"):
    #  TODO: make dir to save all figures,
    #   change files names: remove _ after the first layer.
    """This function draws tree."""
    f = graphviz.Digraph("Simple tree", filename=fname, engine="dot")
    f.attr(rankdir="LR", size="9", title="Markov Chain", labelloc="t")
    f.attr("node", shape="circle", color="crimson", fillcolor="crimson", style="filled")
    f.attr("edge", color="gainsboro")

    for i, weights in enumerate(matrix):
        for j, weight in filter(lambda x: x[-1] > 0, enumerate(weights)):
            f.edge(f"S_{i}", f"S_{j}")

    f.render(view=True)


class Tree:
    """This class creates a tree based on global adjacency matrix and included nodes. """
    def __init__(self, adjacency_matrix, nodes):
        self.adjacency_matrix = adjacency_matrix
        self.__check_matrix()
        self.nodes = nodes
        self.tree = adjacency_matrix[:, nodes][nodes]

    def __str__(self):
        return f'{self.adjacency_matrix} \n {self.tree}'

    def __repr__(self):
        return f'{self.adjacency_matrix}'

    def __check_matrix(self):
        """This private method checks adjacency matrix on basic errors."""
        if self.adjacency_matrix.ndim != 2:
            raise NotTwoDimensionsInAdjacencyMatrix(self.adjacency_matrix.ndim)

        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise DifferentDimensionsOfAdjacencyMatrixError(self.adjacency_matrix.shape)


# Classes for mistakes in creating adjacency matrix
class DifferentDimensionsOfAdjacencyMatrixError(ValueError):
    pass


class NotTwoDimensionsInAdjacencyMatrix(ValueError):
    pass


# Classes for incorrect tree structure
class BadTreeStructure(ValueError):
    pass


class NotOneRootInTree(BadTreeStructure):
    pass


class OnlyRoot(BadTreeStructure):
    pass


class EmptyLayersInTree(BadTreeStructure):
    pass


class Generator:
    # TODO: add method to generate full tree,
    #     add method to convert tree into forest.
    """
    Class to generate tree and convert it to forest by removing random edges and empty nodes.
    """
    def __init__(self, tree_shape, del_level=(np.random.randn(1) / 2 + 0.5)):
        self.shape = tree_shape
        if self.shape[0] != 1:
            raise NotOneRootInTree(f"Number of roots: {self.shape[0]} != 1")
        if self.shape == [1]:
            raise OnlyRoot(f"Tree shape: {self.shape}")
        if any(True for i in self.shape if i == 0):
            raise EmptyLayersInTree(self.shape)

        self.del_level = del_level
        self.total_nodes = sum(self.shape)
        self.tree = self.__tree_maker()

    def __tree_maker(self):
        """This method creates tree."""
        adjacency_matrix = np.zeros([self.total_nodes, self.total_nodes])
        parents = self.shape[0]
        index_children = 1
        index_parent = 0
        for children in self.shape[1:]:
            # Randomise parenthood.
            children_per_parent = np.zeros(parents)
            for child in range(children):
                parent = random.choice([i for i in range(parents)])
                children_per_parent[parent] += 1
            # Add children to parents
            for children_of_parent in children_per_parent:
                for i in range(int(children_of_parent)):
                    adjacency_matrix[index_parent][index_children] = 1
                    index_children += 1
                index_parent += 1
            # Move to next level
            parents = children
        fname = f"{self.shape[0]}_"
        for i in self.shape[1:]:
            fname = fname + "_" + str(i)
        plot_matrix(adjacency_matrix, fname)
        return adjacency_matrix
