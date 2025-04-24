import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graphviz import Digraph


def create_DG(network, engine='networkx', graphviz_format='png', colorize_source_and_drain=True):
    """
    network - np.matrix - матрица смежности (N+1) * (N+1)
    engine - библиотека для построения графа. Варианты: networkx, graphviz
    """
    if engine == 'networkx':
        DG = nx.DiGraph()
        shape = network.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if network[i, j] != 0:
                    DG.add_weighted_edges_from([(i, j + 1, network[i, j])])
        return DG
    elif engine == 'graphviz':
        DG = Digraph(name="QS", format=graphviz_format, graph_attr={'rankdir': 'LR'})  # LR = left to right
        shape = network.shape
        nodes_size = shape[0]

        for n in range(nodes_size+1):
            if colorize_source_and_drain:
                if n == 0 or n == nodes_size:
                    DG.node(name=str(n), label=str(n), color='lightblue2', style='filled')
                else:
                    DG.node(name=str(n), label=str(n))
            else:
                DG.node(name=str(n), label=str(n))


        for i in range(shape[0]):
            for j in range(shape[1]):
                if network[i, j] != 0:
                    DG.edge(str(i), str(j + 1), label=str(network[i, j]))
        return DG
    else:
        print(f"Unknown type of engine {engine}. Please choose from: networkx, graphviz")




def show_DG(DG, engine='networkx', graphviz_format='png'):
    """
    G - граф
    engine - библиотека для построения графа. Варианты: networkx, graphviz
    """
    if engine == 'networkx':
        pos = nx.spring_layout(DG)
        nx.draw_networkx(DG, pos)
        labels = nx.get_edge_attributes(DG, 'weight')
        nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
        plt.show()
    elif engine == 'graphviz':
        DG.render(directory='doctest-output').replace('\\', '/')
        qs = plt.imread(os.path.join(os.getcwd(), 'doctest-output', f'{DG.name}.gv.{graphviz_format}'))
        fig, ax = plt.subplots()
        ax.imshow(qs)
        plt.axis('off')
        fig.tight_layout()
        plt.show()
    else:
        print(f"Unknown type of engine {engine}. Please choose from: networkx, graphviz")


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

    DG = create_DG(R, engine='graphviz')
    # print(os.getcwd())
    show_DG(DG, engine='graphviz')
    # save_in_gephi(DG, "r_example")

