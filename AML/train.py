import torch
from model import GAT, GAS, GATv2Convolution
from dataset import AMLtoGraph
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


print("Available GPU devices:", torch.cuda.device_count())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = AMLtoGraph('/workspace/data', use_rf_features = False)
dataset.process() #process resample
data = dataset[0]
epoch = 500

model = GAT(in_channels=data.num_features, hidden_channels=128, out_channels=1, heads=10) #initial setup
#model = GATv2Convolution(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8, edge_dim = data.edge_attr.shape[1])
#model = GAS(in_channels=data.num_features, hidden_channels=64)
model = model.to(device)
#criterion = torch.nn.BCELoss() #initial setup
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001) #initial setup
pos_weight = dataset.calculate_class_weights()
pos_weight = pos_weight.to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

train_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.train_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.val_mask,
)

threshold = 0.6 #threshold

for i in range(epoch):
    total_loss = 0
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        ground_truth = data.y
        loss = criterion(pred, ground_truth.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    if epoch%10 == 0:
        print(f"Epoch: {i:03d}, Loss: {total_loss:.4f}")
        model.eval()
        acc = 0
        total = 0
        all_preds = [] #ROC
        all_labels = [] #ROC
        with torch.no_grad():
            for test_data in test_loader:
                test_data.to(device)
                pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
                ground_truth = test_data.y
                #binary_preds = (pred > threshold).float() #threshold
                #correct = (binary_preds == ground_truth.unsqueeze(1)).sum().item() #threshold
                correct = (pred == ground_truth.unsqueeze(1)).sum().item()
                total += len(ground_truth)
                acc += correct
                all_preds.append(pred.cpu()) #ROC
                all_labels.append(test_data.y.cpu()) #ROC
            acc = acc/total
            all_preds = torch.cat(all_preds) #ROC
            all_labels = torch.cat(all_labels) #ROC 
            auroc = roc_auc_score(all_labels.numpy(), all_preds.numpy()) #ROC
            binary_preds = (all_preds > 0.5).float() #f1
            f1 = f1_score(all_labels.numpy(), binary_preds.numpy()) #f1
            print('accuracy:', acc)
            print(f'AUROC: {auroc:.4f}')
            print(f'F1-Score: {f1:.4f}')
            cm = confusion_matrix(all_labels.numpy(), binary_preds.numpy())
            recall = cm[1,1] / (cm[1,0] + cm[1,1])
            print(f'Recall: {recall:.4f}')
            print('Confusion Matrix:')
            print(f'[[TN FP]  [{cm[0,0]} {cm[0,1]}]\n [FN TP]]  [{cm[1,0]} {cm[1,1]}]]')

