import os
import torch
import numpy as np
from config import config
from data import data
from model import model
import json

def save_metrics_to_json(metrics_dict, dataset_name):
    file_name = f"{dataset_name}_metrics.json"
    with open(file_name, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {file_name}")

def run_model(model_name, dataset, train_indices, test_indices, args):
    args.model = model_name
    HyperGCN = model.initialise(dataset, args)
    HyperGCN = model.train(HyperGCN, dataset, train_indices, args)
    
    metrics = model.test(HyperGCN, dataset, test_indices, args)
    return metrics

def main():
    args = config.parse()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    model_names = ['gcn', 'gat', 'sage', 'cheb']
    all_metrics = {}

    for model_name in model_names:
        dataset, train, test = data.load(args)
        print("Length of train is", len(train))

        metrics = run_model(model_name, dataset, train, test, args)
        all_metrics[model_name] = metrics

        print(f"Model: {model_name} | accuracy: {metrics['accuracy']:.4f}, "
              f"error: {float(100 * (1 - metrics['accuracy'])):.2f}%")
        print(f"Model: {model_name} | macro_f1: {metrics['macro_f1']:.4f}")
        print(f"Model: {model_name} | weighted_f1: {metrics['weighted_f1']:.4f}")
        print(f"Model: {model_name} | precision: {metrics['precision']:.4f}")
        print(f"Model: {model_name} | recall: {metrics['recall']:.4f}")
        print(f"Model: {model_name} | log_loss: {metrics['log_loss']:.4f}")
        print(f"Model: {model_name} | confusion_matrix: {metrics['confusion_matrix']}")

    # Save all metrics at once
    save_metrics_to_json(all_metrics, args.dataset)

if __name__ == "__main__":
    main()