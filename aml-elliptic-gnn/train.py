import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import json
from torch.utils.tensorboard import SummaryWriter


def train(args, model, data, name):
    """Train a GNN model and return the trained model."""
    writer = SummaryWriter(f'/workspace/runs/{name}') 
    writer.add_graph(model, (data.x, data.edge_index))

    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/Init/{name}', param, 0)
    
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9) #verbose=True
    epochs = args['epochs']
    model.train()

    best_val_loss = float('inf')
    patience = 1000
    epochs_since_best = 0
    all_metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'roc_auc': []
    }
    accuracies = []
    recalls = []

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()


        # Validation
        with torch.no_grad():
          val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
          val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

          # Store accuracy and recall

          y_true = data.y[data.val_mask].cpu().numpy()
          y_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
          y_prob = F.softmax(out[data.val_mask], dim=1).cpu().numpy()[:,1] if out.shape[1] == 2 else None
          recall = recall_score(y_true, y_pred, average='macro')
          precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
        accuracies.append(val_acc)
        recalls.append(recall)
        all_metrics['epoch'].append(epoch)
        all_metrics['train_loss'].append(loss.item())
        all_metrics['val_loss'].append(val_loss.item())
        all_metrics['accuracy'].append(val_acc)
        all_metrics['recall'].append(recall)            
        all_metrics['precision'].append(precision)
        all_metrics['f1'].append(f1)
        all_metrics['roc_auc'].append(roc_auc)
          # Adjust learning rate
        scheduler.step(val_loss)

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            with torch.no_grad():
                features = model.conv1(data.x, data.edge_index)
                writer.add_histogram('Features/conv1_output', features, epoch)


        # Check if early stopping criteria is met
        if epochs_since_best >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    #metrics_path = f"/workspace/metrics_{name}.json"
    #with open(metrics_path, 'w') as f:
        #json.dump(all_metrics, f, indent=2)
    #print(f'Training metrics saved to {metrics_path}')
    writer.close()
    return model, all_metrics, accuracies, recalls

@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out, _ = model((data.x, data.edge_index))
    preds = out[data.test_mask].argmax(dim=1) #visualize
    y_true = data.y[data.test_mask] #visualize
    acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    return y_true.cpu().numpy(), preds.cpu().numpy(), acc
