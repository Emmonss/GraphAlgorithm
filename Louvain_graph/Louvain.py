
import networkx as nx
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
from Louvain_graph.Graph import Graph
import copy

def build_graph_test1():
    G = nx.Graph()
    G.add_nodes_from(['A','B','C','D','E','F'])

    G.add_weighted_edges_from([('A', 'B',5), ('A', 'E',1), ('B', 'C',2),
                               ('A', 'C',4), ('C', 'D',7), ('D', 'F',3),
                               ('E', 'F',8)])

    return G
    # pos = nx.fruchterman_reingold_layout(G)
    # weights = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx(G, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    #
    # plt.show()




class Louvain:
    def __init__(self):
        self.graph=None
        self.iter_index=0
        self.current_graph=None
        self.graph_iter_list = []
        self.init_grapth()

    def init_grapth(self):
        self.current_graph = Graph(build_graph_test1(), iter_num=self.iter_index, nodes_list_iter=None)
        modularity = self.modularity(self.current_graph.graph,self.current_graph.nodes)
        self.current_graph.node_modularity=modularity
        self.current_graph.node_modularity_sum=np.sum(modularity)
        self.graph_iter_list.append(copy.copy(self.current_graph))


        self.iter_one_lovain()
    def iter_one_lovain(self):

        lovain_divergence_max = self.get_max_lovain_comm(self.current_graph)
        if len(lovain_divergence_max.keys())==0:
            #证明已经不能继续缩图了，此时应该停止迭代
            return
        merge_nodes = list(chain.from_iterable(lovain_divergence_max.keys()))
        #
        print(lovain_divergence_max)
        print(merge_nodes)
    def get_max_lovain_comm(self,G):
        lovain_divergence_max = {}
        is_noded = []
        for node in self.current_graph.nodes:
            max_modularity,node_max=-1,None
            relation_node = [v for u, v, wt in G.edges(node, data="weight", default=1) if v not in is_noded]
            for r_node in relation_node:
                r_q = self.modular_divergence(G,r_node,node)
                if r_q>0 and r_q>max_modularity:
                    node_max=r_node
                    max_modularity=r_q
            if not node_max==None:
                lovain_divergence_max[(node,node_max)]=max_modularity
            is_noded.append(node)
        return lovain_divergence_max



    def modular_divergence(self,G,community,node_add,weight="weight"):
        ki_in = sum(wt for u, v, wt in G.edges(node_add, data=weight, default=1) if v in community)
        ki = sum(wt for u, v, wt in G.edges(node_add, data=weight, default=1))
        Etot = sum(wt for u, v, wt in G.edges(community, data=weight, default=1))
        m = G.all_weight
        # print(ki_in,ki,Etot)
        # print((ki_in-ki*Etot/m))
        return (ki_in-ki*Etot/m)/(2*m)

    #计算社区模块度
    def modularity(self,G,community, weight="weight", resolution=1):
        directed_flag = G.is_directed()

        if directed_flag:
            out_degree = dict(G.out_degree(weight=weight))
            in_degree = dict(G.in_degree(weight=weight))
            m = sum(out_degree.values())
            norm = 1/m**2

        else:
            out_degree = in_degree = dict(G.degree(weight=weight))
            deg_sum = sum(out_degree.values())
            m = deg_sum/2
            norm = 1/deg_sum**2

        def community_contribution(node_list):
            comm = set(node_list)
            L_c = sum(wt for u,v,wt in G.edges(comm,data=weight,default=1) if v in comm)
            out_degree_sum = sum(out_degree[u] for u in comm)
            in_degree_sum = sum(in_degree[u] for u in comm) if directed_flag else out_degree_sum
            return (L_c/m - resolution*out_degree_sum*in_degree_sum*norm)*m

        return list(map(community_contribution,community))


from pprint import pprint
if __name__ == '__main__':
    l = Louvain()

