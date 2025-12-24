import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from model import utils 



class HyperGCN(nn.Module):
    def __init__(self, V, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dataset == 'citeseer': power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.mediators)        
        else:
            reapproximate = True
            structure = E
            
        self.layers = nn.ModuleList([utils.HyperGraphConvolution(h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.depth
        self.structure, self.m = structure, args.mediators

    def forward(self, H):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        features = []
        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            features.append(H)
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)
        
        return F.log_softmax(H, dim=1), features

class HyperGAT(nn.Module):
    def __init__(self, V, E, X, args):
        super(HyperGAT, self).__init__()
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dataset == 'citeseer': power = l - i + 4
            h.append(2**power)
        h.append(c)

        self.layers = nn.ModuleList([utils.HyperGraphAttention(h[i], h[i+1], cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.depth
        self.structure, self.m = E, args.mediators

    def forward(self, H):
        do, l, m = self.do, self.l, self.m
        features = []
        for i, hidden in enumerate(self.layers):
            H = hidden(self.structure, H, m)
            features.append(H)
            if i < l - 1:
                H = F.dropout(F.elu(H), do, training=self.training)
        
        return F.log_softmax(H, dim=1), features

class HyperSAGE(nn.Module):
    def __init__(self, V, E, X, args):

        super(HyperSAGE, self).__init__()
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()

        h = [d]
        for i in range(l - 1):
            power = l - i + 2
            if args.dataset == 'citeseer': 
                power = l - i + 4
            h.append(2 ** power)
        h.append(c)

        self.layers = nn.ModuleList([utils.HyperGraphSAGE(h[i], h[i + 1], cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.depth
        self.structure, self.m = E, args.mediators

    def forward(self, H):
        """
        an l-layer SAGE
        """
        do, l, m = self.do, self.l, self.m
        features = []
        for i, layer in enumerate(self.layers):
            H = layer(self.structure, H, m)
            features.append(H)
            if i < l - 1:
                H = F.dropout(F.relu(H), do, training=self.training)
        
        return F.log_softmax(H, dim=1), features
    


class ChebNet(nn.Module):
    def __init__(self, V, E, X, args, cheb_k=5):
        super(ChebNet, self).__init__()
        
        d, l, c = args.d, args.depth, args.c
        self.K = cheb_k  
        
        h = [d]
        for i in range(l - 1):
            power = l - i + 2
            if args.dataset == 'citeseer': 
                power = l - i + 4
            h.append(2 ** power)
        h.append(c)

       
        if args.fast:
            self.structure = utils.Laplacian(V, E, X, args.mediators)        
        else:
            self.structure = E
            
        self.layers = nn.ModuleList([utils.ChebGraphConvolution(h[i], h[i + 1], self.K, self.structure) for i in range(l)])
        self.do = args.dropout
         

    def forward(self, H):
        features = []
        for i, layer in enumerate(self.layers):
            H = layer(self.structure, H)  
            features.append(H)
            if i < len(self.layers) - 1:
                H = F.dropout(H, self.do, training=self.training)
                H = F.relu(H)

        return F.log_softmax(H, dim=1), features