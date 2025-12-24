import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_results(df, save_path):
    labels = df['model'].to_numpy()
    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    f1_micro = df['F1 Micro AVG'].to_numpy()
    roc_auc = df['ROC AUC'].to_numpy()

    x = np.arange(len(labels))  
    width = 0.15  

    _, ax = plt.subplots(figsize=(20, 7))

    ax.bar(x - 2*width, precision, width, label='Precision', color='#83f27b')
    ax.bar(x - width, recall, width, label='Recall', color='#f27b83')
    ax.bar(x, f1, width, label='F1', color='#f2b37b')
    ax.bar(x + width, f1_micro, width, label='Micro AVG F1', color='#7b8bf2')
    ax.bar(x + 2*width, roc_auc, width, label='ROC AUC', color='#b37bf2')

    ax.set_ylabel('Value')
    ax.set_title('Metrics by Classifier')
    ax.set_xticks(x)  
    ax.set_xticklabels(labels=labels, rotation=90)
    ax.legend(loc="lower left")
    ax.set_yticks(np.arange(0, 1.1, 0.1))  

    plt.grid(True)
    plt.tight_layout()  
    plt.savefig(save_path)
    plt.show()

def aggregate_plot(df, save_path):
    labels = df['model'].to_numpy()

    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    maf1 = df['F1 Micro AVG'].to_numpy()
    roc_auc = df['ROC AUC'].to_numpy()  

    x = np.arange(len(labels))  
    width = 0.55  
    fig, ax = plt.subplots(figsize=(6, 10))

    ax.bar(x, f1, width, label='F1 Score', color='#f2b37b')
    ax.bar(x, maf1, width, label='M.A. F1 Score', color='#7b8bf2', bottom=f1)
    ax.bar(x, precision, width, label='Precision', color='#83f27b', bottom=maf1 + f1)
    ax.bar(x, recall, width, label='Recall', color='#f27b83', bottom=maf1 + f1 + precision)
    ax.bar(x, roc_auc, width, label='ROC AUC', color='#b37bf2', bottom=maf1 + f1 + precision + recall)

    ax.set_ylabel('Value (0-1)')
    ax.set_title('Final Metrics by Classifier')
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_yticks(np.arange(0, 4, 0.1))
    ax.set_xticklabels(labels=labels)
    ax.legend()

    plt.xticks(rotation=90)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_radar_chart(model_names, metrics, data_path):
    num_metrics = len(metrics[0]) 
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#f5f5f5')  
    ax.set_facecolor('#ffffff') 
    colors = sns.color_palette() 
    
    for i, model in enumerate(model_names):
        radar_metric = np.concatenate([metrics[i], [metrics[i][0]]])
        
        ax.fill(angles, radar_metric, color=colors[i], alpha=0.02)
        ax.plot(angles, radar_metric, color=colors[i], linewidth=2, label=model)  

    plt.xticks(angles[:-1], ['Precision', 'Recall', 'F1 Score', 'M.A. F1 Score', 'ROC AUC']) 
    plt.title('Model Performance Comparison', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

metrics_file_path = './data/metrics.csv' 
df = pd.read_csv(metrics_file_path)

plot_results(df, './data/model_performance.png')
aggregate_plot(df, './data/aggregate_performance.png')

model_names = df['model'].to_numpy()
metrics_values = df[['Precision', 'Recall', 'F1', 'F1 Micro AVG', 'ROC AUC']].to_numpy()

data_path = './data' 
if not os.path.exists(data_path):
    os.makedirs(data_path)

plot_radar_chart(model_names, metrics_values, data_path)