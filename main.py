import numpy as np
from time import strftime
from basic_classes import Tree, Generator, Solver
import sys


def main():

    # rnd_adj_m = np.random.rand(3, 3)
    # nodes = np.array([0, 1])
    # rnd_tree = Tree(rnd_adj_m, nodes)
    # print(rnd_tree)
    gen = Generator([1, 4, 4, 4, 4, 4, 4, 6], 0, 0.7)
    nodes = []
    # for tree in gen.forest:
    #     nodes.append(tree.get_nodes())
    solver = Solver(gen.matrix, gen.forest)
    final_tree = solver.solve()
    print("Diameter of the graph after the solution:", len(final_tree.find_dim_chain()))
    # gen = Generator([1, 2, 2])


if __name__ == "__main__":
    main()
