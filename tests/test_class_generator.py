import pytest
from ..basic_classes import Generator, BadTreeStructure
import numpy as np


@pytest.mark.parametrize("gen_good_tree", [
    [[1, 2, 4], 1],
    [[1, 2, 4, 8, 16], 1],
    [[1, 3, 1, 6, 2], 1],
])
def test__gen_good_tree(gen_good_tree):
    gen = Generator(gen_good_tree[0])
    assert gen.tree.all() != gen_good_tree[1]


@pytest.mark.parametrize("gen_bad_structure_tree", [
    [[1], 1],  # We can't see tree. No children.
    [[2, 2], 1],  # Two roots.
    [[1, 0, 4], 1],  # Empty layers.
    [[1, 2, 3], 1],  # This should fail cause of correct structure
])
def test__gen_bad_structure_tree(gen_bad_structure_tree, capsys):
    with pytest.raises(BadTreeStructure) as e_info:
        _ = Generator(gen_bad_structure_tree[0])





