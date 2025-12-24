import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os

import numpy as np
import time
import datetime
import path
import shutil
import json
import config
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

args = config.parse()


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'


#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'/workspace/{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}' )
print(f"Output directory: {out_dir}")

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

 

### configure logger 
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)



# load data
from data import data
from prepare import * 


test_accs = []
best_val_accs, best_test_accs = [], []

resultlogger.info(args)

# load data
X, Y, G = fetch_data(args)
metrics_list = []
epochs = []  # 存储所有 epoch 的列表
best_test_accs_list = []  # 存储每个 epoch 的最佳测试准确率
accuracies = []  # 存储每个 epoch 的准确率
macro_precisions = []  # 存储每个 epoch 的宏平均精确率
macro_recalls = []  # 存储每个 epoch 的宏平均召回率
macro_f1s = []  # 存储每个 epoch 的宏平均 F1 分数

for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # load data
    args.split = run
    _, train_idx, test_idx = data.load(args)
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx  = torch.LongTensor(test_idx ).cuda()

    # model 
    model, optimizer = initialise(X, Y, G, args)


    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()


    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    baselogger.info(counter)
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

    best_test_acc, test_acc, Z = 0, 0, None   
    all_features = [] 
    y_true = Y[test_idx].cpu().numpy() 

    model.eval()
    with torch.no_grad():
        Z, initial_features = model(X, True)
        all_features.append(initial_features.detach().cpu())
        initial_filename = f'features_epoch_0_{args.model_name}_{args.dataset}.pt'
        torch.save(initial_features, out_dir / initial_filename)
        print(f"Initial features saved to: {out_dir / initial_filename}")

    for epoch in range(args.epochs):
        # train
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z, features = model(X, True)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])

        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        
        # eval
        model.eval()
        Z, features = model(X, True)
        train_acc= accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])

        y_pred = torch.argmax(Z[test_idx], dim=1).cpu().numpy()
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # log acc
        best_test_acc = max(best_test_acc, test_acc)
        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')
        #all_features.append(features.detach().cpu())
        if (epoch + 1) % 10 == 0:
            all_features = features.detach().cpu()
            filename = f'features_epoch_{epoch + 1}_{args.model_name}_{args.dataset}.pt'
            torch.save(all_features, out_dir / filename)
            print(f"Features saved to: {out_dir / filename}")
        epochs.append(epoch + 1) 
        best_test_accs_list.append(best_test_acc)
        accuracies.append(test_acc)
        macro_precisions.append(macro_precision)
        macro_recalls.append(macro_recall)
        macro_f1s.append(macro_f1)
    #all_features = torch.cat(all_features, dim=0)
    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)

metrics = {
    "epoch": epochs,
    "best_test_acc": best_test_accs_list,
    "accuracy": accuracies,
    "macro_precision": macro_precisions,
    "macro_recall": macro_recalls,
    "macro_f1": macro_f1s
}
with open(out_dir / 'metrics.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)
resultlogger.info(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
filename = f'features_{args.model_name}_{args.dataset}.pt'
model_filename = f'{args.model_name}_{args.dataset}_model.pth'
torch.save(model.state_dict(), out_dir / model_filename)
print(f"Model saved to: {out_dir / model_filename}")
