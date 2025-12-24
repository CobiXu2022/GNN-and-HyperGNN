from model import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm
from model import utils
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix
import numpy as np



def train(HyperGCN, dataset, T, args):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	T: training indices
	args: arguments

	returns:
	the trained model
    """    
    
    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()
    
    X, Y = dataset['features'], dataset['labels']
    class_weights = HyperGCN.get('class_weights') 
    all_features = [] 
    for epoch in tqdm(range(args.epochs)):

        optimiser.zero_grad()
        Z, features = hypergcn(X)
        all_features.append(features[-1]) 
        if class_weights is not None:
            loss = F.nll_loss(Z[T], Y[T], weight=class_weights)
        else:
            loss = F.nll_loss(Z[T], Y[T])

        loss.backward()
        optimiser.step()
    torch.save(all_features, f'training_features_{args.dataset}_{args.model}.pt') 
    torch.save(Z, f'output_{args.dataset}_{args.model}.pt') 
    print(f"Saved features with shape: {[f.shape for f in all_features]}")
    HyperGCN['model'] = hypergcn
    return HyperGCN



def test(HyperGCN, dataset, t, args):
    hypergcn = HyperGCN['model']
    hypergcn.eval()
    X, Y = dataset['features'], dataset['labels']
    
    with torch.no_grad():
        Z, features = hypergcn(X)        
        torch.save(features[-1], f'test_features_{args.dataset}_{args.model}.pt') 
        Y_true = Y[t].cpu().numpy()
        Y_pred = Z[t].max(1)[1].cpu().numpy()
        
        logits = Z[t]
        max_logits = logits.max(dim=1, keepdim=True).values
        stable_logits = logits - max_logits  
        probabilities = torch.exp(stable_logits) / torch.exp(stable_logits).sum(dim=1, keepdim=True)

        probabilities = probabilities.cpu().numpy()

        probabilities = np.nan_to_num(probabilities)

    acc = accuracy(Z[t], Y[t])
    macro_f1 = f1_score(Y_true, Y_pred, average='macro')
    weighted_f1 = f1_score(Y_true, Y_pred, average='weighted')
    precision = precision_score(Y_true, Y_pred, average='weighted')
    recall = recall_score(Y_true, Y_pred, average='weighted')
    logloss = log_loss(Y_true, probabilities)
    conf_matrix = confusion_matrix(Y_true, Y_pred)

    return {
        'accuracy': acc.item(),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'log_loss': logloss,
        'confusion_matrix': conf_matrix.tolist(),
    }

def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns: 
    accuracy
    """
    
    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy



def initialise(dataset, args):
    """
    initialises GCN, optimiser, normalises graph, and features, and sets GPU number
    
    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    args: arguments

    returns:
    a dictionary with model details (hypergcn, optimiser)    
    """
    
    HyperGCN = {}
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']

    # hypergcn and optimiser
    args.d, args.c = X.shape[1], Y.shape[1]
    if args.model == 'gcn':
        hypergcn = networks.HyperGCN(V, E, X, args)
    elif args.model == 'gat':
        hypergcn = networks.HyperGAT(V, E, X, args)
    elif args.model == 'sage':
        hypergcn = networks.HyperSAGE(V, E, X, args)
    elif args.model == 'cheb':
        hypergcn = networks.ChebNet(V, E, X, args)
    else:
        raise ValueError("Unknown model type: {}".format(args.model))
    
    if args.optimizer == 'adam':
        optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)
    elif args.optimizer == 'sgd':
        optimiser = optim.SGD(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)
    elif args.optimizer == 'adadelta':
        optimiser = optim.Adadelta(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(args.optimizer))


    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    # cuda
    args.Cuda = args.cuda and torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        X, Y = X.cuda(), Y.cuda()

    # update dataset with torch autograd variable
    dataset['features'] = Variable(X)
    dataset['labels'] = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser

    class_weights = dataset.get('class_weights')
    if class_weights is not None and args.Cuda:
        class_weights = class_weights.cuda()
    HyperGCN['class_weights'] = class_weights
    return HyperGCN



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(axis=1)).flatten()
    di = np.zeros_like(d, dtype=np.float32)
    nonzero_mask = d != 0
    di[nonzero_mask] = 1. / d[nonzero_mask]
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
