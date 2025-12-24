import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, roc_auc_score
import torch
import seaborn as sns

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config():
    with open("/workspace/config.yaml", "r") as config:
        args = AttributeDict(yaml.safe_load(config))
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    return args

def accuracy(pred_y, y):
    """Calculate accuracy"""
    return ((pred_y == y).sum() / len(y)).item()

def compute_metrics(model, name, data, df):

  _, y_predicted = model((data.x, data.edge_index))[0].to("cpu").max(dim=1)
  data = data.to("cpu")

  prec_ill,rec_ill,f1_ill,_ = precision_recall_fscore_support(data.y[data.test_mask], y_predicted[data.test_mask], average='binary', pos_label=0)
  f1_micro = f1_score(data.y[data.test_mask], y_predicted[data.test_mask], average='micro')
  roc_auc = roc_auc_score(data.y[data.test_mask].numpy(), y_predicted[data.test_mask].numpy())

  m = {'model': name, 'Precision': np.round(prec_ill,3), 'Recall': np.round(rec_ill,3), 'F1': np.round(f1_ill,3),
   'F1 Micro AVG':np.round(f1_micro,3), 'ROC AUC': np.round(roc_auc, 3) }

  return m

def plot_results(df, save_path):

    labels = df['model'].to_numpy()
    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    f1_micro = df['F1 Micro AVG'].to_numpy()
    roc_auc = df['ROC AUC'].to_numpy()

    x = np.arange(len(labels))
    width = 0.15
    _, ax = plt.subplots(figsize=(28, 7))
    ax.bar(x - width/2, precision, width, label='Precision',color='#83f27b')
    ax.bar(x + width/2, recall, width, label='Recall',color='#f27b83')
    ax.bar(x - (3/2)*width, f1, width, label='F1',color='#f2b37b')
    ax.bar(x + (3/2)*width, f1_micro, width, label='Micro AVG F1',color='#7b8bf2')
    ax.bar(x + (2)*width, roc_auc, width, label='ROC AUC', color='#b37bf2')

    ax.set_ylabel('value')
    ax.set_title('Metrics by classifier')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,1,0.05))
    ax.set_xticklabels(labels=labels)
    ax.legend(loc="lower left")

    plt.grid(True)

    plt.savefig(save_path)
    plt.show()

def aggregate_plot(df, save_path):

    labels = df['model'].to_numpy()

    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    maf1 = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))  # the label locations
    width = 0.55  # the width of the bars
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.bar(x, f1, width, label='F1 Score',color='#f2b37b')
    ax.bar(x , maf1, width, label='M.A. F1 Score',color='#7b8bf2',bottom=f1)
    ax.bar(x, precision, width, label='Precision',color='#83f27b',bottom=maf1 + f1)
    ax.bar(x, recall, width, label='Recall',color='#f27b83',bottom=maf1 + f1 + precision)

    ax.set_ylabel('value 0-1')
    ax.set_title('Final metrics by classifier')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,4,0.1))
    ax.set_xticklabels(labels=labels)
    ax.legend()

    plt.xticks(rotation=90)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_accuracy_recall(accuracies, recalls, model_name, data_path):
    plt.figure(figsize=(12, 8))
    plt.plot([epoch * 100 for epoch in range(len(accuracies))], accuracies, label='Accuracy', marker='o', markersize=3, linewidth=1)
    plt.plot([epoch * 100 for epoch in range(len(recalls))], recalls, label='Recall', marker='x', markersize=3, linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(f'Accuracy and Recall Over Epochs for {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(data_path, f'{model_name}_accuracy_recall_epochs.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, data_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(data_path, f'{model_name}_confusion_matrix.png'))


def plot_heatmap(accuracies, recalls, model_names, data_path):
    max_length = max(len(acc) for acc in accuracies + recalls)

    accuracies_padded = [np.pad(acc, (0, max_length - len(acc)), 'constant', constant_values=np.nan) for acc in accuracies]
    recalls_padded = [np.pad(rec, (0, max_length - len(rec)), 'constant', constant_values=np.nan) for rec in recalls]

    heatmap_data = np.vstack([accuracies_padded, recalls_padded])
    yticklabels = [f"{name} Accuracy" for name in model_names] + [f"{name} Recall" for name in model_names]
    plt.figure(figsize=(20, 12))
    sns.heatmap(heatmap_data, annot=False, fmt='.2f', cmap='YlGnBu', 
                xticklabels=[i * 100 for i in range(1, max_length + 1)], 
                yticklabels=yticklabels)

    plt.title('Accuracy and Recall Heatmap')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.savefig(os.path.join(data_path, 'accuracy_recall_heatmap.png'))
    plt.close()

def plot_radar_chart(model_names, metrics, data_path):
    num_metrics = len(metrics[0])
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    metrics = np.array(metrics)
    metrics = np.concatenate((metrics, metrics[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw=dict(polar=True))

    fig.patch.set_facecolor('#f5f5f5')  
    ax.set_facecolor('#ffffff') 
    colors = sns.color_palette() 
    for i, model in enumerate(model_names):
        ax.fill(angles, metrics[i], color=colors[i], alpha=0.02) 
        ax.plot(angles, metrics[i], color=colors[i], linewidth=2, label=model)  

    plt.xticks(angles[:-1], ['Precision', 'Recall', 'F1 Score', 'ROC AUC']) 
    plt.title('Model Performance Comparison')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(data_path, 'radar_chart.png'))
    plt.close()



class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__