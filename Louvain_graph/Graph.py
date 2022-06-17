import networkx as nx
import numpy as np

def build_graph_test1():
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])

    G.add_weighted_edges_from([('A', 'B', 5), ('A', 'E', 1), ('B', 'C', 2),
                               ('A', 'C', 4), ('C', 'D', 7), ('D', 'F', 3),
                               ('E', 'F', 8)])

    return G

class Graph:
    def __init__(self,graph,**kwargs):
        self.graph = graph
        self.iter_index = kwargs['iter_num']
        self.nodes_list_iter = kwargs['nodes_list_iter']
        self.get_graph_attrs()

    def get_graph_attrs(self):
        self.edges = self.graph.edges
        self.edges_weight = nx.get_edge_attributes(self.graph, "weight")
        self.nodes = self.graph.nodes
        self.all_weight = np.sum(list(self.edges_weight.values()))

        self.node_modularity = None
        self.node_modularity_sum=-1

        # print(self.edges)
        # print(self.nodes)
        # print(self.edges_weight)
        # print(self.all_weight)
        # pass



if __name__ == '__main__':
    g = build_graph_test1()
    g1 = Graph(g,iter_num=0,nodes_list_iter=None)