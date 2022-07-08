import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import os,json
import random
from itertools import chain

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
        self.merge_node_prob = kwargs['merge_node_prob']
        self.node_group = None
        self.get_graph_attrs()

    def get_graph_attrs(self):
        self.edges = self.graph.edges
        self.edges_weight = nx.get_edge_attributes(self.graph, "weight")
        self.nodes = self.graph.nodes
        self.all_weight = np.sum(list(self.edges_weight.values()))

        self.node_modularity = None
        self.node_modularity_sum=-1


        # pass
    def pprint_graph(self):
        print('='*30)
        print("edges:{}".format(self.edges))
        print("nodes:{}".format(self.nodes))
        print("edges_weight:{}".format(self.edges_weight))
        print("all_weight:{}".format(self.all_weight))
        print("node_modularity:{}".format(self.node_modularity))
        print("node_modularity_sum:{}".format(self.node_modularity_sum))
        print("node_group:{}".format(self.node_group))
        print("nodes_list_iter:{}".format(self.nodes_list_iter))

    def graph_save(self,save_root_path,png_root_path):
        nx.write_gpickle(self.graph, os.path.join(save_root_path,'lovain_iter_{}.gpickle'.format(self.iter_index)))
        save_info = {
            'nodes':str(self.nodes),
            'edges_weight':str(self.edges_weight),
            'all_weight':str(self.all_weight),
            'node_modularity':str(self.node_modularity),
            'node_modularity_sum':str(self.node_modularity_sum),
            'node_group':str(self.node_group)
        }
        # print(save_info)
        with open(os.path.join(save_root_path,'lovain_iter_{}.json'.format(self.iter_index)), 'w') as fw:
            js = json.dumps(save_info)
            fw.write(js)
            fw.close()

        if len(self.nodes)<=50:
            # print('=' * 30)
            plt.clf()
            pos = nx.fruchterman_reingold_layout(self.graph)
            weights = nx.get_edge_attributes(self.graph, "weight")
            # print("self.node_group:{}".format(self.node_group))
            color_map = [1]*len(self.get_2_dimen_len(self.node_group))
            color_index = 1
            # print("color_map_init:{}".format(color_map))
            for item in self.node_group:
                if not isinstance(item,str):
                    for i in item:
                        color_map[list(self.nodes).index(i)]=color_index
                else:
                    color_map[list(self.nodes).index(item)]=color_index
                # break
                color_index+=1

            # print("color_map:{}".format(color_map))
            nx.draw_networkx(self.graph, pos, with_labels=True, node_color=color_map)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=weights)
            #添加颜色区分
            #
            plt.savefig(os.path.join(png_root_path,"lovain_iter_{}.png".format(self.iter_index)),format='PNG')

    def get_2_dimen_len(self,list2):
        res = []
        for item1 in list2:
            if isinstance(item1,tuple) or isinstance(item1,list):
                for item2 in item1:
                    res.append(item2)
            else:
                res.append(item1)
        return res



if __name__ == '__main__':
    g = build_graph_test1()
    # g1 = Graph(g,iter_num=0,nodes_list_iter=None)