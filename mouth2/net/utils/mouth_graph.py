import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_unique_nodes(self, connections):
        unique_nodes = set()
        for cn in connections:
            unique_nodes.add(cn[0])
            unique_nodes.add(cn[1])

        return sorted(unique_nodes)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'mmpose':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 9), (0, 17), (1, 9), (1, 10), (1, 2), (1, 3), (1, 4), (2, 10), (2, 11), (2, 13), (2, 3), (3, 13), (3, 14), (3, 4), (4, 14), (4, 15), (4, 5), (5, 15), (5, 19), (6, 12), (6, 13), (6, 11), (6, 17), (6, 18), (6, 7), (6, 8), (
                7, 8), (7, 18), (8, 18), (8, 19), (8, 13), (8, 16), (9, 17), (9, 12), (9, 10), (10, 11), (10, 12), (11, 12), (11, 13), (12, 17), (13, 14), (13, 16), (14, 15), (14, 16), (15, 16), (15, 19), (16, 19), (17, 18), (17, 20), (18, 19), (18, 20), (19, 20)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.unique_nodes = self.get_unique_nodes(neighbor_link)
        # elif layout=='customer settings'
        #     pass
        elif layout == 'mmpose_mouth_11':
            self.num_node = 11
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(3, 4), (3, 10), (5, 10), (8, 9), (0, 2), (1, 6), (6, 8), (4, 5), (5, 6), (5, 9), (
                0, 1), (9, 10), (1, 2), (0, 4), (2, 7), (1, 5), (6, 7), (4, 10), (5, 8), (0, 3), (1, 4), (2, 6), (7, 8)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.unique_nodes = self.get_unique_nodes(neighbor_link)
        # elif layout=='customer settings'
        #     pass
        elif layout == 'mmpose_mouth_23':
            self.num_node = 23
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(6, 18), (3, 4), (18, 20), (12, 13), (21, 22), (4, 15), (0, 2), (5, 16), (8, 9), (0, 5), (14, 22), (8, 18), (1, 6), (17, 18), (17, 21), (2, 8), (13, 14), (11, 20), (16, 22), (18, 19), (6, 17), (4, 5), (5, 6), (12, 21), (3, 15), (14, 15), (0, 1), (9, 10), (
                1, 2), (0, 4), (9, 19), (2, 7), (1, 5), (10, 11), (17, 20), (19, 20), (10, 20), (6, 7), (15, 16), (13, 22), (6, 16), (15, 22), (7, 18), (20, 21), (4, 16), (3, 14), (0, 3), (11, 12), (2, 9), (1, 7), (8, 19), (17, 22), (11, 21), (10, 19), (7, 8), (13, 21), (16, 17)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.unique_nodes = self.get_unique_nodes(neighbor_link)
        elif layout == 'mmpose_mouth_8':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5),
                             (5, 6), (6, 7), (1, 7), (0, 1)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.unique_nodes = self.get_unique_nodes(neighbor_link)
        elif layout == 'mmpose_mouth_12':
            self.num_node = 12
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                             (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (1, 11), (0, 1)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.unique_nodes = self.get_unique_nodes(neighbor_link)
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            # print(self.A)
            # print(self.unique_nodes)
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD