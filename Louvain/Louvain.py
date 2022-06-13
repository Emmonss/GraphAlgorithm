
import networkx as nx
from matplotlib import pyplot as plt


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
        self.all_weight = 0
        self.edge_attr = 0
        self.build_graph()

    def build_graph(self):
        self.graph = build_graph_test1()
        self.res = self.modularity(self.graph, ['A', 'B', 'C','D','E','F'])
        print(list(self.res),sum(list(self.res)))
        self.res = self.modularity(self.graph,[['A','B'],'C','D','E','F'])
        print(list(self.res),sum(self.res))

        self.res = self.modularity(self.graph, [['A', 'B'], ['C', 'D'], ['E', 'F']])
        print(list(self.res), sum(list(self.res)))

        self.weights = nx.get_edge_attributes(self.graph, "weight")
        # for item in self.weights:
        #     print(item[0],item[1],self.weights[item])
        # print(self.weights)


    #计算整体图的模块度
    def modularity(self,G,community, weight="weight", resolution=1):
        directed_flag = G.is_directed()

        if directed_flag:
            out_degree = dict(G.out_degree(weight=weight))
            in_degree = dict(G.in_degree(weight=weight))
            m = sum(out_degree.values())
            norm = 1/m**2

        else:
            out_degree = in_degree = dict(G.degree(weight=weight))
            # print(out_degree)
            deg_sum = sum(out_degree.values())
            m = deg_sum/2
            norm = 1/deg_sum**2


        def community_contribution(node_list):
            # print('='*50)
            comm = set(node_list)
            # print([(u, v, wt) for u, v, wt in G.edges(comm, data=weight, default=1)])
            # print([(u,v,wt) for u,v,wt in G.edges(comm,data=weight,default=1) if v in comm])
            L_c = sum(wt for u,v,wt in G.edges(comm,data=weight,default=1) if v in comm)
            # print(comm,m,L_c)
            out_degree_sum = sum(out_degree[u] for u in comm)
            in_degree_sum = sum(in_degree[u] for u in comm) if directed_flag else out_degree_sum
            # print(out_degree_sum,in_degree_sum)
            return (L_c/m - resolution*out_degree_sum*in_degree_sum*norm)*m

        return map(community_contribution,community)



from pprint import pprint
if __name__ == '__main__':
    l = Louvain()

