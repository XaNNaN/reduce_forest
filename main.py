import numpy as np

from basic_classes import Tree


def main():
    rnd_adj_m = np.random.rand(5, 5)
    rnd_tree = Tree(rnd_adj_m)
    print(rnd_tree)



if __name__ == "__main__":
    main()
