import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_scatter import scatter 


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        


    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'

class HyperGraphAttention(Module):
    """
    HyperGraph Attention layer.
    """

    def __init__(self, in_features, out_features, cuda=True):
        super(HyperGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(2 * out_features, 1))  # Attention weights
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.a.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    def forward(self, structure, H, m):
        H_prime = torch.mm(H, self.W)  
    
        node_to_index = {node: idx for idx, node in enumerate(structure.keys())}
    
        indices = []
        edge_attention = []
    
        for i, neighbors in structure.items(): 
            idx_i = node_to_index[i]
            neighbor_indices = []
        
            for j in neighbors:
                if isinstance(j, int):
                    idx_j = j  
                else:
                    if j not in node_to_index:
                        print(f"Warning: {j} not found in node_to_index") 
                        continue
                    idx_j = node_to_index[j]
            
                neighbor_indices.append(idx_j)
                indices.append((idx_i, idx_j))

            if neighbor_indices:
                neighbors_tensor = H_prime[neighbor_indices]
                edge_attention.append(F.leaky_relu((H_prime[idx_i] * neighbors_tensor).sum(dim=1), negative_slope=0.2))

        edge_attention = torch.cat(edge_attention) if edge_attention else torch.tensor([], device=H_prime.device)
        row_indices, col_indices = zip(*indices)
    
        attention = torch.sparse_coo_tensor(
            torch.tensor([row_indices, col_indices], device=H_prime.device),
            edge_attention,
            size=(H_prime.size(0), H_prime.size(0)),
            device=H_prime.device
        )
    
        attention = sparse_softmax(attention)
        H_out = SparseMM.apply(attention, H_prime) + self.bias
        return H_out

    def _single_attention(self, h_i, h_j):
        return F.leaky_relu(torch.dot(h_i, h_j), negative_slope=0.2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class HyperGraphSAGE(Module):
    """
    HyperGraph SAGE layer.
    """

    def __init__(self, in_features, out_features, cuda=True):
        super(HyperGraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m):
        H_prime = torch.mm(H, self.W)

        edge_attention = []
        indices = []
        node_to_index = {node: idx for idx, node in enumerate(structure.keys())}

        for i, neighbors in structure.items():
            for j in neighbors:
                idx_i = node_to_index[i]
                if isinstance(j, int):
                    idx_j = j
                else:
                    if j not in node_to_index:
                        print(f"Warning: {j} not found in node_to_index")
                        continue
                    idx_j = node_to_index[j]
                edge_attention.append(F.leaky_relu((H_prime[idx_i] * H_prime[idx_j]).sum(), negative_slope=0.2))
                indices.append((idx_i, idx_j))

        edge_attention = torch.tensor(edge_attention, device=H_prime.device)
        row_indices, col_indices = zip(*indices)

        attention = torch.sparse_coo_tensor(
            torch.tensor([row_indices, col_indices], device=H_prime.device),
            edge_attention,
            size=(H_prime.size(0), H_prime.size(0)),
            device=H_prime.device
        )

        attention = sparse_softmax(attention)

        H_out = SparseMM.apply(attention, H_prime) + self.bias
        return H_out


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ChebGraphConvolution(nn.Module):
    """
    ChebNet layer that uses Chebyshev polynomials for graph convolution.
    """

    def __init__(self, in_features, out_features, K, structure):
        super(ChebGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K  
        self.structure = structure 
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.bias)

    def chebyshev_polynomial(self, L):
        """Compute Chebyshev polynomials up to order K."""
        T_k = [torch.eye(L.size(0)).to(L.device), L]
        for i in range(2, self.K + 1):
            T_k.append(2 * L @ T_k[-1] - T_k[-2])
        return T_k

    def forward(self, structure, H):
        device = H.device
    
        if isinstance(structure, dict):
            n_nodes = H.size(0)
            with torch.no_grad():  
                L = Laplacian(n_nodes, structure, H.cpu().detach().numpy(), m=False)
                L = L.to(device)
        else:
            L = structure
    
        n = L.size(0)
        indices = torch.stack([torch.arange(n, device=device), 
                             torch.arange(n, device=device)])
        values = torch.ones(n, device=device)
        eye = torch.sparse_coo_tensor(indices, values, (n, n), device=device)
        
        L_normalized = 2 * L - eye
        
        H_out = torch.mm(H, self.W)
        if self.K >= 1:
            T_prev = H
            T_curr = torch.sparse.mm(L_normalized, H)
            H_out = H_out + torch.mm(T_curr, self.W)
            
            for k in range(2, self.K+1):
                with torch.no_grad():
                    T_next = 2 * torch.sparse.mm(L_normalized, T_curr) - T_prev
                del T_prev
                torch.cuda.empty_cache()
                H_out = H_out + torch.mm(T_next, self.W)
                T_prev, T_curr = T_curr, T_next
                del T_next
                torch.cuda.empty_cache()
        return H_out + self.bias

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)



def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights



def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A



def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def sparse_softmax(sparse_mat):
    """
    Apply softmax on a sparse matrix.
    
    Arguments:
    sparse_mat -- A sparse tensor.
    
    Returns:
    A sparse tensor after applying softmax.
    """
    sparse_mat = sparse_mat.coalesce()
    exp_values = torch.exp(sparse_mat.values())
    row_indices = sparse_mat.indices()[0]
    sum_exp = scatter(exp_values, row_indices, dim=0, dim_size=sparse_mat.size(0), reduce="sum")
    norm_values = exp_values / (sum_exp[row_indices] + 1e-10)
    return torch.sparse_coo_tensor(
        sparse_mat.indices(),
        norm_values,
        sparse_mat.size(),
        device=sparse_mat.device
    )
