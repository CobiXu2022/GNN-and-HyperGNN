import torch
import json
import os
from pathlib import Path


dir_path = '/home/zmxu/Desktop/ly/UniGNN/runs/UniSAGE_2_cocitation_pubmed/seed_1/'

for filename in os.listdir(dir_path):
    if filename.endswith('.pt'): 
        pt_path = os.path.join(dir_path, filename)
        
        try:
            tensor_data = torch.load(pt_path)
            
            if torch.is_tensor(tensor_data):
                numpy_data = tensor_data.numpy()
                data_to_save = numpy_data.tolist()
            else:
                data_to_save = tensor_data  
            
            json_filename = filename.replace('.pt', '.json')
            json_path = os.path.join(dir_path, json_filename)
            
            with open(json_path, 'w') as json_file:
                json.dump(data_to_save, json_file)
            
            print(f"Successfully converted {filename} to {json_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All .pt files have been processed.")