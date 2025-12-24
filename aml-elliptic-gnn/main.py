import warnings
import torch
import pandas as pd
import utils as u
import os
from loader import load_data, data_to_pyg
from train import train, test
from models import models
from argparse import ArgumentParser
from models.custom_gat.model import GAT

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

models_dir = '/workspace/saved_models'
os.makedirs(models_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("="*50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)

print("Graph data loaded successfully")
print("="*50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print ("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)

models_to_train = {
    'GCN Convolution (tx)': models.GCNConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GCN Convolution (tx+agg)': models.GCNConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GIN Convolution (tx)': models.GINConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GIN Convolution (tx+agg)': models.GINConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GAT Convolution (tx)': models.GATConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GAT Convolution (tx+agg)': models.GATConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GTC Convolution (tx)': models.GTConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GTC Convolution (tx+agg)': models.GTConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'SAGE Convolution (tx)': models.SAGEConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'SAGE Convolution (tx+agg)': models.SAGEConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'Chebyshev Convolution (tx)': models.ChebyshevConvolution(args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev Convolution (tx+agg)': models.ChebyshevConvolution(args, [1, 2], data.num_features, args.hidden_units).to(args.device),
    'GATv2 Convolution (tx)': models.GATv2Convolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GATv2 Convolution (tx+agg)': models.GATv2Convolution(args, data.num_features, args.hidden_units).to(args.device)
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("="*50)

performance_metrics = []
all_metrics_list = []
accuracies_list = []
recalls_list = []
model_names = []

model_list = list(models_to_train.items())

metrics_list = []

for i in range(0, len(model_list), 2):

    (name, model) = model_list[i]
    data_noAgg = data_noAgg.to(args.device)

    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    model, all_metrics, accuracies_noAgg, recalls_noAgg = train(args, model, data_noAgg, name)
    all_metrics_list.append({name: all_metrics}) 
    model_filename = f"{name.split('(')[0].strip().replace(' ', '_')}_tx.pt" 
    model_path = os.path.join(models_dir, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': name,
        'input_features': data_noAgg.num_features,
        'hidden_units': args.hidden_units_noAgg,
        #'args': args
    }, model_path)
    print(f"Model saved to {model_path}")

    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    y_true, y_pred, acc= test(model, data_noAgg)
    #u.plot_confusion_matrix(y_true, y_pred, name, data_path)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    print('-'*50)
    metrics = u.compute_metrics(model, name, data_noAgg, compare_illicit)
    if isinstance(metrics, dict):
        metrics = pd.DataFrame([metrics])  
    compare_illicit = pd.concat([compare_illicit, metrics], ignore_index=True)

    precision = metrics['Precision'].values[0]  
    recall = metrics['Recall'].values[0]
    f1_score = metrics['F1'].values[0]
    roc_auc = metrics['ROC AUC'].values[0]
    performance_metrics.append([precision, recall, f1_score, roc_auc])

    model_accuracies = []
    model_recalls = []
    
    for epoch in range(len(accuracies_noAgg)):
        if epoch % 100 == 0:  
            model_accuracies.append(accuracies_noAgg[epoch])
            model_recalls.append(recalls_noAgg[epoch])

    for epoch in range(len(accuracies_noAgg)):
        if epoch % 40 == 0:  
            metrics_list.append({
                'Model': name,
                'Epoch': epoch,
                'Accuracy': accuracies_noAgg[epoch],
                'Recall': recalls_noAgg[epoch]
            })
    
    accuracies_list.append(model_accuracies)
    recalls_list.append(model_recalls)
    model_names.append(name)
    #u.plot_accuracy_recall(model_accuracies, model_recalls, name, data_path)

    (name, model) = model_list[i + 1]
    data = data.to(args.device)
    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    model, all_metrics, accuracies, recalls = train(args, model, data, name)
    all_metrics_list.append({name: all_metrics}) 
    model_filename = f"{name.split('(')[0].strip().replace(' ', '_')}_tx_agg.pt" 
    model_path = os.path.join(models_dir, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': name,
        'input_features': data.num_features,
        'hidden_units': args.hidden_units,
        #'args': args
    }, model_path)
    print(f"Model saved to {model_path}")

    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    y_true, y_pred, acc= test(model, data) 
    #u.plot_confusion_matrix(y_true, y_pred, name, data_path)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    metrics = u.compute_metrics(model, name, data, compare_illicit)
    if isinstance(metrics, dict):
        metrics = pd.DataFrame([metrics])  
    compare_illicit = pd.concat([compare_illicit, metrics], ignore_index=True)

    precision = metrics['Precision'].values[0]  
    recall = metrics['Recall'].values[0]
    f1_score = metrics['F1'].values[0]
    roc_auc = metrics['ROC AUC'].values[0]
    performance_metrics.append([precision, recall, f1_score, roc_auc])

    model_accuracies = []
    model_recalls = []
    
    for epoch in range(len(accuracies)):
        if epoch % 100 == 0:  
            model_accuracies.append(accuracies[epoch])
            model_recalls.append(recalls[epoch])

    for epoch in range(len(accuracies)):
        if epoch % 40 == 0:  
            metrics_list.append({
                'Model': name,
                'Epoch': epoch,
                'Accuracy': accuracies[epoch],
                'Recall': recalls[epoch]
            })
    
    accuracies_list.append(model_accuracies)
    recalls_list.append(model_recalls)
    model_names.append(name)
    #u.plot_accuracy_recall(model_accuracies, model_recalls, name, data_path)
    print('-'*50)
    

compare_illicit.to_csv(os.path.join(data_path, 'metrics.csv'), index=False)
print('Results saved to metrics.csv')

metrics_storage = pd.DataFrame(metrics_list)
metrics_storage.to_csv(os.path.join(data_path, 'acc_rec.csv'), index=False)
print('Metrics saved to acc_rec.csv')

#u.plot_radar_chart(model_names, performance_metrics, data_path)

#u.plot_heatmap(accuracies_list, recalls_list, model_names, data_path)
#u.plot_results(compare_illicit, save_path=os.path.join(data_path, 'model_performance.png'))

#u.aggregate_plot(compare_illicit, save_path=os.path.join(data_path, 'aggregate_performance.png'))

