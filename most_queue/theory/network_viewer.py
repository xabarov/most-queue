import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def create_DG(network):
    """
    network - np.matrix - матрица смежности (N+1) * (N+1)
    """
    DG = nx.DiGraph()
    shape = network.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if network[i, j] != 0:
                DG.add_weighted_edges_from([(i, j + 1, network[i, j])])

    return DG


def show_DG(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def save_in_gephi(G, save_name):
    nx.readwrite.gexf.write_gexf(G, save_name + ".gexf")


if __name__ == '__main__':
    R = np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 0.4, 0.6, 0, 0, 0],
        [0, 0, 0, 0.6, 0.4, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    DG = create_DG(R)
    show_DG(DG)
    save_in_gephi(DG, "r_example")
