import pytest
from ..basic_classes import Tree, NotTwoDimensionsInAdjacencyMatrix
import numpy as np


@pytest.mark.parametrize("bad_tree_init", [
    [2],
    [2, 1, 3],
])
def test__bad_tree_init(bad_tree_init):
    with pytest.raises(NotTwoDimensionsInAdjacencyMatrix):
        tree = np.random.rand(*bad_tree_init)
        Tree(tree, [i for i in range(len(bad_tree_init))])


@pytest.mark.parametrize("good_tree_init", [
    [2, 2],
])
def test__good_tree_init(good_tree_init):
    tree = np.random.rand(*good_tree_init)
    tree = Tree(tree, [i for i in range(len(good_tree_init))])
    assert tree.adjacency_matrix.shape == tuple(good_tree_init)
