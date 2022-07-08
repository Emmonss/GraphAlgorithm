
import networkx as nx
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
from Louvain_graph.Graph import Graph
import copy,os,random

save_root_path='./graph_iter'
png_root_path = './png_iter_limit_50'

def build_graph_test1():
    G = nx.Graph()
    G.add_nodes_from(['A','B','C','D','E','F'])

    G.add_weighted_edges_from([('A', 'B',5), ('A', 'E',1), ('B', 'C',2),
                               ('A', 'C',4), ('C', 'D',7), ('D', 'F',3),
                               ('E', 'F',8)])

    return G

def build_graph_test2(node_num=20,edge_prob=0.8):
    nodes = ['node_{}'.format(i) for i in range(node_num)]
    edges = []
    i = 0
    while (i < node_num):
        j = 0
        while (j < i):
            probability = random.random()  # 生成随机小数
            if (probability > edge_prob):  # 如果大于p
                edges.append(('node_{}'.format(i),'node_{}'.format(j),random.randint(5,20)))
            j += 1
        i += 1
    G = nx.Graph()
    G.add_nodes_from(nodes)  # 添加节点
    G.add_weighted_edges_from(edges)
    return G

class Louvain:
    def __init__(self):
        self.graph=None
        self.current_graph=None
        self.graph_iter_list = []
        self.init_grapth()

    def init_grapth(self):
        self.clear_save()
        self.iter_index = 0
        self.current_graph = Graph(build_graph_test2(node_num=15),
                                   iter_num=self.iter_index,
                                   nodes_list_iter=None,
                                   merge_node_prob=1.0)

        modularity = self.modularity(self.current_graph.graph,self.current_graph.nodes)
        self.current_graph.node_modularity=modularity
        self.current_graph.node_modularity_sum=np.sum(modularity)
        self.graph_iter_list.append(copy.copy(self.current_graph))

        self.lovain_algorith()




    def lovain_algorith(self,prob_limit=0.1):
        flag = True
        while flag:
            # print(self.iter_index)
            merge_nodes, new_nodes, weighted_edges, merge_node_prob, nodes_list_iter,flag = self.iter_one_lovain()
            if flag and merge_node_prob>prob_limit:
                self.iter_index += 1
                self.current_graph = Graph(self.build_new_graph(new_nodes,weighted_edges),
                                           iter_num=self.iter_index, nodes_list_iter=nodes_list_iter,
                                           merge_node_prob=merge_node_prob)
                modularity = self.modularity(self.current_graph.graph, self.current_graph.nodes)
                self.current_graph.node_modularity = modularity
                self.current_graph.node_modularity_sum = np.sum(modularity)
                self.current_graph.node_modularity = modularity
                #添加到list
                self.graph_iter_list.append(copy.copy(self.current_graph))
            else:
                flag = False

        self.get_cluster()

        # for item in self.graph_iter_list:
        #     item.pprint_graph()
        # # save processing
        for grah in self.graph_iter_list:
            grah.graph_save(save_root_path,png_root_path)

    def clear_save(self):
        grap_list = os.listdir(save_root_path)
        for item in grap_list:
            os.remove(os.path.join(save_root_path,item))
        png_list = os.listdir(png_root_path)
        for item in png_list:
            os.remove(os.path.join(png_root_path, item))

    def get_cluster(self):
        self.graph_iter_list[-1].node_group = self.graph_iter_list[-1].nodes
        cluster_upper = self.graph_iter_list[-1].node_group
        # print('cluster_upper:{}'.format(cluster_upper))
        for index in range(len(self.graph_iter_list) - 2, -1, -1):
            cluster = []
            cluster_dict = self.graph_iter_list[index+1].nodes_list_iter
            # print('cluster_dict:{}'.format(cluster_dict))
            for item in cluster_upper:
                if isinstance(item, tuple) or isinstance(item, list):
                    c1 = []
                    for i in item:
                        cls = cluster_dict[i]
                        if isinstance(cls, str):
                            c1.append(cls)
                        else:
                            for f in cls:
                                c1.append(f)
                    cluster.append(c1)
                elif isinstance(item, str):
                    cluster.append(cluster_dict[item])
            # print("cluster:{}".format(cluster))
            self.graph_iter_list[index].node_group = cluster
            cluster_upper = cluster
        #
        # for item in self.graph_iter_list:
        #     item.pprint_graph()


    def iter_one_lovain(self):
        lovain_divergence_max = self.get_max_lovain_comm(self.current_graph)
        if len(lovain_divergence_max.keys())==0:
            #没有节点相加的模块度大于0
            #证明已经不能继续缩图了，此时应该停止迭代
            return [],[],[],0,{},False
        merge_nodes = list(lovain_divergence_max.keys())
        merge_nodes_exist = list(chain.from_iterable(lovain_divergence_max.keys()))
        #添加无需融合的节点
        rest_node = list(set(self.current_graph.nodes)-set(merge_nodes_exist))
        merge_node_prob = float((len(self.current_graph.nodes)-len(rest_node))/len(self.current_graph.nodes))
        merge_nodes.extend(rest_node)
        new_nodes = ['iter_{}_node_{}'.format(self.iter_index,node_index)
                          for node_index in range(1,len(merge_nodes)+1)]

        nodes_list_iter = {}
        for index in range(len(new_nodes)):
            nodes_list_iter[new_nodes[index]] = merge_nodes[index]
        # print("nodes_list_iter:{}".format(nodes_list_iter))

        weighted_edges =[]
        for index1 in range(len(merge_nodes)):
            for index2 in range(index1+1,len(merge_nodes)):
                new_weight = sum([wt for u, v, wt in
                        self.current_graph.edges(merge_nodes[index1], data="weight", default=1)
                        if v in merge_nodes[index2]])
                weighted_edges.append((new_nodes[index1],new_nodes[index2],new_weight))
        return merge_nodes,new_nodes,weighted_edges,merge_node_prob,nodes_list_iter,True


    def build_new_graph(self,node,edge_weight):
        G = nx.Graph()
        G.add_nodes_from(node)
        G.add_weighted_edges_from(edge_weight)
        return G

    def get_max_lovain_comm(self,G):
        lovain_divergence_max = {}
        is_noded = []
        for node in self.current_graph.nodes:
            if node in is_noded:
                continue
            max_modularity,node_max=-1,None
            relation_node = [v for u, v, wt in G.edges(node, data="weight", default=1) if v not in is_noded]
            for r_node in relation_node:
                r_q = self.modular_divergence(G,r_node,node)
                if r_q>0 and r_q>max_modularity:
                    node_max=r_node
                    max_modularity=r_q
            if not node_max==None:
                lovain_divergence_max[(node,node_max)]=max_modularity
            #如果存在，则挑模块度增益最大的那个节点加入的那个社区，二者全部都加入到已识别中
            is_noded.extend([node,node_max])
        return lovain_divergence_max

    def modular_divergence(self,G,community,node_add,weight="weight"):
        ki_in = sum(wt for u, v, wt in G.edges(node_add, data=weight, default=1) if v in community)
        ki = sum(wt for u, v, wt in G.edges(node_add, data=weight, default=1))
        Etot = sum(wt for u, v, wt in G.edges(community, data=weight, default=1))
        m = G.all_weight
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
            if isinstance(node_list,str):
                comm = set([node_list])
            elif isinstance(node_list,list):
                comm = set(node_list)
            else:
                comm = set([node_list])
            L_c = sum(wt for u,v,wt in G.edges(comm,data=weight,default=1) if v in comm)
            out_degree_sum = sum(out_degree[u] for u in comm)
            in_degree_sum = sum(in_degree[u] for u in comm) if directed_flag else out_degree_sum
            return (L_c/m - resolution*out_degree_sum*in_degree_sum*norm)*m

        return list(map(community_contribution,community))


from pprint import pprint
if __name__ == '__main__':
    l = Louvain()

