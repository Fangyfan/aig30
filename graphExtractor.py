import numpy as np
from numpy import linalg as LA
import numpy as np
import dgl
import torch


# def symmetricLaplacian(abc):
#     numNodes = abc.numNodes()
#     L = np.zeros((numNodes, numNodes))
#     print("numNodes", numNodes)
#     for nodeIdx in range(numNodes):
#         aigNode = abc.aigNode(nodeIdx)
#         degree = float(aigNode.numFanouts())
#         if (aigNode.hasFanin0()):
#             degree += 1.0
#             fanin = aigNode.fanin0()
#             L[nodeIdx][fanin] = -1.0
#             L[fanin][nodeIdx] = -1.0
#         if (aigNode.hasFanin1()):
#             degree += 1.0
#             fanin = aigNode.fanin1()
#             L[nodeIdx][fanin] = -1.0
#             L[fanin][nodeIdx] = -1.0
#         L[nodeIdx][nodeIdx] = degree
#     return L


# def symmetricLapalacianEigenValues(abc):
#     L = symmetricLaplacian(abc)
#     print("L", L)
#     eigVals = np.real(LA.eigvals(L))
#     print("eigVals", eigVals)
#     return eigVals


def extract_dgl_graph(abc) -> dgl.DGLGraph:
    G = dgl.DGLGraph()
    G.add_nodes(abc.numNodes())
    edge0 = [
        (abc.aigNode(nodeIdx).fanin0(), nodeIdx)
        for nodeIdx in range(G.num_nodes())
        if abc.aigNode(nodeIdx).hasFanin0()
    ]
    edge1 = [
        (abc.aigNode(nodeIdx).fanin1(), nodeIdx)
        for nodeIdx in range(G.num_nodes())
        if abc.aigNode(nodeIdx).hasFanin1()
    ]
    edge0.extend(edge1)
    u = [e[0] for e in edge0]
    v = [e[1] for e in edge0]
    G.add_edges(u, v)
    features = [
        [
            1.0 if abc.aigNode(nodeIdx).nodeType() == nodeType else 0.0
            for nodeType in range(6)
        ]
        for nodeIdx in range(G.num_nodes())
    ]
    features = torch.tensor(features)
    G = dgl.add_self_loop(G)
    return G
