import os
import torch
import pandas as pd
from argparse import ArgumentParser
from loader import load_data, data_to_pyg
import utils as u

models_dir = '/workspace/saved_models'
os.makedirs(models_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("=" * 50)

print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)

model_name = 'GCN_Convolution_tx'  
model_path = os.path.join(models_dir, f"{model_name}.pt")
model = MyModel(input_features=features.shape[1], hidden_units=64)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 

input_tensor = torch.tensor(features, dtype=torch.float32)

predictions_list = []  

with torch.no_grad(): 
    outputs = model(input_tensor)
    predictions = torch.sigmoid(outputs) 
    predicted_classes = (predictions > 0.5).int()  
    predictions_list.extend(predicted_classes.cpu().numpy()) 

output_file = 'predictions.csv'
pd.DataFrame(predictions_list, columns=['Predicted Class']).to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")