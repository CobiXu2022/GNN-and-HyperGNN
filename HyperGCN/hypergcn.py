# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
from config import config
args = config.parse()



# seed
import os, torch, numpy as np
torch.manual_seed(args.seed)
np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data
from data import data
dataset, train, test = data.load(args)
print("length of train is", len(train))



# # initialise HyperGCN
from model import model
HyperGCN = model.initialise(dataset, args)



# train and test HyperGCN
HyperGCN = model.train(HyperGCN, dataset, train, args)
metrics = model.test(HyperGCN, dataset, test, args)
print(f"accuracy: {metrics['accuracy']:.4f}", ", error:", float(100*(1-metrics['accuracy'])))
print(f"macro_f1: {metrics['macro_f1']:.4f}")
print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
