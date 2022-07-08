import networkx as nx
from matplotlib import pyplot as plt
from itertools import chain

d0 = ['iter_0_node_1','iter_0_node_2','iter_0_node_3']
d1 = {'iter_0_node_1': ('A', 'B'), 'iter_0_node_2': ('C', 'D'), 'iter_0_node_3': ('E', 'F')}
d2 = {'A':('1','2'),
      'B':('3','4'),
      'C':('5','6'),
      'D':('7','8'),
      'E':'9',
      'F':'10'}

d3 = {'1':('11','12'),
      '2':('23','24'),
      '3':('35','36'),
      '4':('47','48'),
      '5':'59',
      '6':'610',
      '7':'759',
      '8':'859',
      '9':'959',
      '10':'1059',}

d = [d3,d2,d1,d0]
res = {}
if __name__ == '__main__':
    # dicts = d0
    # for index in range(len(d)-2,-1,-1):
    #     cluster = []
    #     print("dict:{}".format(dicts))
    #     print(d[index])
    #     for item in dicts:
    #         if isinstance(item,tuple) or isinstance(item,list):
    #             c1 = []
    #             for i in item:
    #                 cls = d[index][i]
    #                 if isinstance(cls,str):
    #                     c1.append(cls)
    #                 else:
    #                     for f in cls:
    #                         c1.append(f)
    #             cluster.append(c1)
    #         elif isinstance(item,str):
    #             cluster.append(d[index][item])
    #     print("cluster:{}".format(cluster))
    #     dicts = cluster

    import random
    node_num=20
    edge_prob = 0.8
    nodes = [str(i) for i in range(node_num)]
    edges = []
    i = 0
    while (i < node_num):
        j = 0
        while (j < i):
            probability = random.random()  # 生成随机小数
            if (probability > edge_prob):  # 如果大于p
                edges.append((str(i),str(j),random.randint(1,100)))
            j += 1
        i += 1
    print(edges)
    G = nx.Graph()
    G.add_nodes_from(nodes)  # 添加节点
    G.add_weighted_edges_from(edges)

    pos = nx.fruchterman_reingold_layout(G)
    weights = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

    plt.show()
