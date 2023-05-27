import numpy as np
import random
import graphviz
# Working with images
from time import strftime
import os
import sys


def plot_matrix(matrix, im_path, fname="plot", dim_chains=[], center=[]):
    """This function draws tree."""
    __ = type(dim_chains[0])
    if type(dim_chains[0]) != list:
        dim_chains = [dim_chains]
    f = graphviz.Graph("Simple tree", filename=fname, engine="dot")
    f.attr(rankdir="LR", size="9", title="Markov Chain", labelloc="t")
    f.attr("node", shape="circle", color="darkorange2", fillcolor="darkorange2", style="filled")
    f.attr("edge", color="gainsboro")

    for i, weights in enumerate(matrix):
        for j, weight in filter(lambda x: x[-1] > 0, enumerate(weights)):
            if j > i:
                if i in center:
                    f.node(f"S_{i}", color="red", fillcolor="red")
                _ = [True if i in dim_chain and j in dim_chain else False for dim_chain in dim_chains]
                if any(_):
                    f.attr("edge", color="red")
                    f.edge(f"S_{i}", f"S_{j}")
                else:
                    f.attr("edge", color="gainsboro")
                    f.edge(f"S_{i}", f"S_{j}")
        if sum(weights) == 0:
            f.node(f"S_{i}")

    f.render(directory=im_path, view=True)


class Tree:
    """This class creates a tree based on global adjacency matrix and included nodes. """

    def __init__(self, adjacency_matrix, nodes):
        self.adjacency_matrix = adjacency_matrix
        self.__check_matrix()
        self.nodes = nodes
        self.tree = adjacency_matrix[:, nodes][nodes]
        self.center, self.leaves = self.__find_center()
        self.chain = self.__find_dim_chain()

    def __str__(self):
        return f"{self.chain}"

    def __repr__(self):
        return f"{self.chain}"

    def appcenter(self, small_tree):
        self.nodes += small_tree.nodes
        self.adjacency_matrix[self.center][small_tree.center] = 1
        self.adjacency_matrix[small_tree.center][self.center] = 1
        self.tree = self.adjacency_matrix[:, self.nodes][self.nodes]
        self.chain = self.__find_dim_chain()

    def get_nodes(self):
        return self.nodes

    def find_dim_chain(self):
        return self.__find_dim_chain()

    def __check_matrix(self):
        """This private method checks adjacency matrix on basic errors."""
        if self.adjacency_matrix.ndim != 2:
            raise NotTwoDimensionsInAdjacencyMatrix(self.adjacency_matrix.ndim)

        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise DifferentDimensionsOfAdjacencyMatrixError(self.adjacency_matrix.shape)

    def __find_center(self):
        adj_matrix = self.tree.copy()  # Matrix
        roots = [i for i in range(adj_matrix.shape[0])]
        leaves = []

        # i = roots[0]
        # idx = 0
        new_roots = []
        tmp_roots = []
        flag = 1
        first_leaves = []
        center = self.nodes[roots[0]]
        while len(roots) > 2:
            for i in roots:
                if i in tmp_roots:
                    new_roots.append(i)
                elif np.sum(adj_matrix[i][i:]) == 0:  # No children
                    parent = adj_matrix[i].argmax()
                    adj_matrix[parent][i] = 0
                    adj_matrix[i][parent] = 0

                    leaves.append(i)
                    tmp_roots.append(parent)
                elif np.sum(adj_matrix[i][0:i]) == 0 and np.sum(adj_matrix[i][i:]) == 1:  # No parent & 1 child
                    child = adj_matrix[i].argmax()
                    adj_matrix[child][i] = 0
                    adj_matrix[i][child] = 0

                    leaves.append(i)
                    tmp_roots.append(child)
                else:
                    new_roots.append(i)

            if len(new_roots) != 0:
                roots = new_roots.copy()
            else:
                roots = tmp_roots.copy()
            center = self.nodes[roots[0]]

            new_roots = []
            tmp_roots = []
            if flag == 1:
                first_leaves = leaves.copy()
            flag = 0

        return center, first_leaves

    def __find_dim_chain(self):
        start = self.nodes.index(self.center)
        candidates = self.leaves
        chain = []
        traces = []

        unvisited_nodes = [i for i in range(self.tree.shape[0])]
        unvisited_nodes.remove(start)

        neighbors = [idx for idx, el in enumerate(self.tree[start]) if el == 1 and idx in unvisited_nodes]
        for i in neighbors:
            traces.append([i])

        trace = None
        to_visit = neighbors.copy()
        nodes_to_visit = {el: traces[idx] for idx, el in enumerate(to_visit)}
        tmp_nodes = {}

        while unvisited_nodes:
            for node, trace in nodes_to_visit.items():
                if node in candidates:
                    unvisited_nodes.remove(node)
                else:
                    neighbors = [idx for idx, el in enumerate(self.tree[node]) if el == 1 and idx in unvisited_nodes]
                    for neighbor in neighbors:
                        traces.append(trace + [neighbor])
                        tmp_nodes[neighbor] = traces[-1]
                    traces.remove(trace)
                    unvisited_nodes.remove(node)
            nodes_to_visit = tmp_nodes.copy()
            tmp_nodes.clear()

        traces = sorted(traces, key=lambda x: len(x), reverse=True)
        if traces != []:
            way_1 = traces[0]
        elif trace is not None:
            way_1 = trace
        else:
            way_1 = []
        if type(way_1) == int:
            way_1 = list(way_1)
        way_2 = []
        for trace in traces[1:]:
            _ = [el != way_1[idx] for idx, el in enumerate(trace)]
            if all(_):
                way_2 = trace
                break

        if type(way_2) == int:
            way_1 = list(way_2)
        chain = chain + way_2 + way_1 + [start]
        return list(np.array(self.nodes)[chain])

    def get_center(self):
        return self.center

    def get_chain(self):
        return self.chain


class Solver:
    def __init__(self, matrix, forest):
        self.adj_matrix = matrix
        self.forest = forest

        dir_name = strftime("%Y_%B_%d_%Hh_%Mm_%Ss")
        if sys.argv[0][-7:] == "main.py":
            self.images_path = f"./images/{dir_name}"
        else:
            self.images_path = f"../images/{dir_name}"

    def solve(self):
        trees_sorted = sorted(self.forest, key=lambda tree: len(tree.get_chain()), reverse=True)
        main_tree = trees_sorted[0]

        for idx, tree in enumerate(trees_sorted[1:]):
            main_tree.appcenter(tree)
            self.forest.remove(tree)
            self.plot_forest(idx)

        return main_tree

    def plot_forest(self, i):
        dim_chains = []
        centers = []
        for tree in self.forest:
            dim_chains.append(tree.get_chain())
            centers.append(tree.get_center())

        plot_matrix(self.forest[0].adjacency_matrix, self.images_path, fname=f"forest_{i}", dim_chains=dim_chains, center=centers)


class Generator:
    # TODO: add method to generate full tree???
    #       Add method to convert tree into forest.
    #       It is possible to del edge if vertex has more then 1 edge.
    #       In this case we also need fun to calcute min redius
    #       Центр графа- это вершина с минимальным радиусом.
    #       Диаметр графа должен оставаться прежним. Находим радиусы для каждой вершины, по ним определяем центр,
    #       Фиксируем диаметральную цепь, удаеляем только те рёбра, коих нет в диам. цепи
    """
    Class to generate tree and convert it to forest by removing random edges and empty nodes.
    """

    def __init__(self, tree_shape, del_level=(np.random.randn(1) / 2 + 0.5), weight=1):
        self.shape = tree_shape
        if self.shape[0] != 1:
            raise NotOneRootInTree(f"Number of roots: {self.shape[0]} != 1")
        if self.shape == [1]:
            raise OnlyRoot(f"Tree shape: {self.shape}")
        if any(True for i in self.shape if i == 0):
            raise EmptyLayersInTree(self.shape)

        self.del_level = del_level
        self.total_nodes = sum(self.shape)

        # Создаём дерево
        self.tree = Tree(self.__tree_maker(weight), [i for i in range(self.total_nodes)])
        print("Diameter of the graph before the solution", len(self.tree.get_chain()))

        # Отрисовываем дерево, попутно найдя диаметр и центр
        fname = f"{self.shape[0]}"
        for i in self.shape[1:]:
            fname = fname + "_" + str(i)

        self.dim_chain = self.tree.get_chain()
        self.center = self.tree.get_center()
        print(type(self.center))
        plot_matrix(self.tree.tree, self.img_path, fname, self.dim_chain, [self.center])

        self.matrix, trees_by_nodes = self.__delete_nodes()
        self.forest = []
        # plot_matrix(self.matrix, self.img_path, fname="tree_2", dim_chains=self.dim_chain)
        for tree in trees_by_nodes:
            self.forest.append(Tree(self.matrix, tree))
        dim_chains = []
        centers = []
        for tree in self.forest:
            dim_chains.append(tree.get_chain())
            centers.append(tree.get_center())

        plot_matrix(self.matrix, self.img_path, fname="tree_3", dim_chains=dim_chains, center=centers)

    def __delete_nodes(self):
        del_lvl = self.del_level
        shape = self.tree.tree.shape[0]
        chain = self.dim_chain

        leaves = []
        for idx, node in enumerate(self.tree.tree):
            if np.sum(node) == 1 and idx not in chain:
                leaves.append(idx)

        max_nodes_to_del = self.tree.tree.shape[0] - len(self.dim_chain) - len(leaves)
        if max_nodes_to_del < 1:
            print(f"There is no nodes to delete. Try to create another tree.")
            return
        nodes_to_del = int(max_nodes_to_del * self.del_level)

        print(f"At max. {nodes_to_del} will be deleted. Total possible nodes to delete: {max_nodes_to_del}.")

        Bset = frozenset(chain + leaves)
        nodes = [i for i in range(0, self.tree.tree.shape[0])]
        nodes = [item for item in nodes if item not in Bset]  # C = A - B
        # nodes = [i for i in range(0, self.tree.shape[0])] - chain
        random.shuffle(nodes)
        del_nodes = []
        del_counter = 0
        tmp_tree_2 = self.tree.tree.copy()

        trees = [[i for i in range(0, self.tree.tree.shape[0])]]

        if not nodes_to_del:
            return tmp_tree_2, trees

        for i in range(nodes_to_del):
            parent = tmp_tree_2[nodes[0]].argmax()
            neighbors = [int(i) for i in self.tree.tree[nodes[0]] if i == 1 and i != parent]
            if sum(tmp_tree_2[nodes[0]][nodes[0]:]) > 0 \
                    and sum(tmp_tree_2[parent][parent:]) > 1 :
                    # and all([sum(self.tree.tree[neighbor][neighbor:]) > 0 for neighbor in neighbors]):
                del_nodes.append(nodes[0])
                node = nodes[0]
                parent = self.tree.tree[node].argmax()
                tmp_tree_2[parent][node] = 0
                tmp_tree_2[node][parent] = 0
                for idx, tree in enumerate(trees):
                    if node in tree:
                        parent = idx
                trees[parent].remove(node)
                trees.append([node])
                neighbors = [idx for idx, el in enumerate(tmp_tree_2[node]) if el == 1]
                while neighbors != []:
                    tmp_neighbors = neighbors.copy()
                    for neighbor in neighbors:
                        trees[-1].append(neighbor)
                        tmp_neighbors.remove(neighbor)
                        trees[parent].remove(neighbor)
                        tmp_neighbors += [idx for idx, el in enumerate(tmp_tree_2[neighbor]) if
                                          el == 1 and idx not in trees[-1]]
                    neighbors = tmp_neighbors.copy()

                del_counter += 1
            nodes.pop(0)
        if del_counter == 1:
            print(f"{del_counter} edge was deleted.")
        elif del_counter > 1:
            print(f"{del_counter} edges were deleted.")
        else:
            print(f"No edges were deleted.")

        indexes = nodes + chain
        indexes.sort()
        indexes = np.array(indexes)
        tmp_tree = self.tree.tree[np.ix_(indexes, indexes)]

        # plot_matrix(tmp_tree, self.img_path, fname="matrix_after_del")
        # plot_matrix(tmp_tree_2, self.img_path, fname="tree_after_del", dim_chain=self.dim_chain)

        return tmp_tree_2, trees

    def __find_dim_chain_reserve(self):  # This method should be recreated incorrect result
        chain = []
        idx = sum(self.shape) - 1

        # Первая ветка
        while idx != 0:
            chain.append(idx)
            idx = self.tree[idx].argmax()

        # Ищем вторую ветку
        idx = sum(self.shape) - 2
        temp_chain = []
        while idx != 0:
            parent = self.tree[idx].argmax()
            temp_chain.append(idx)
            if parent == 0:
                if idx not in chain:
                    idx = 0
                    chain = chain + temp_chain
                else:
                    idx = idx - 1
                    temp_chain = []
            idx_2 = idx
            # Проверяем всех родителей и всех пра, и пра-пра, и т. д.
            while parent != 0:
                parent = self.tree[idx_2].argmax()
                # Если мы у корня, то добавляем временную цепь к основной, мы нашли диаметральную цепь
                if parent == 0:
                    chain = chain + temp_chain
                    idx = 0
                # Если родители у веток совпадают, то идём к следующему кандидату
                if parent in chain:
                    idx = idx - 1
                    parent = 0
                    temp_chain = []
                else:
                    temp_chain.append(parent)
                    idx_2 = parent.copy()

        chain.append(0)
        return chain

    def __tree_maker(self, weight=1):
        """This method creates tree.
            weight задаёт вес вершине, получившей ребёнка. При значении выше 1 эта врешина получит других детей
            с большей вероятностью. И наоборот.
        """
        adjacency_matrix = np.zeros([self.total_nodes, self.total_nodes])
        parents = self.shape[0]
        index_children = 1
        index_parent = 0
        for children in self.shape[1:]:
            # Randomise parenthood.
            children_per_parent = np.zeros(parents)
            weights = np.ones(parents)
            for child in range(children):
                parent = random.choices([i for i in range(parents)], weights=weights, k=1)[0]
                weights[parent] = weight
                children_per_parent[parent] += 1
            # Add children to parents
            for children_of_parent in children_per_parent:
                for i in range(int(children_of_parent)):
                    adjacency_matrix[index_parent][index_children] = 1
                    adjacency_matrix[index_children][index_parent] = 1
                    index_children += 1
                index_parent += 1
            # Move to next level
            parents = children
        # Plotting
        dir_name = strftime("%Y_%B_%d_%Hh_%Mm_%Ss")
        if sys.argv[0][-7:] == "main.py":
            images_path = f"./images/{dir_name}"
        else:
            images_path = f"../images/{dir_name}"
        try:
            os.mkdir(images_path)
        except OSError as error:
            print(error)
        self.img_path = images_path

        return adjacency_matrix


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
