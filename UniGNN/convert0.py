import torch
import json
import os
pt_path = '/home/zmxu/Desktop/ly/UniGNN/runs/UniGIN_2_cocitation_pubmed/seed_1/features_epoch_0_UniGIN_pubmed.pt'

json_path = pt_path.replace('.pt', '.json')

try:
    tensor_data = torch.load(pt_path, map_location='cpu')
    
    if isinstance(tensor_data, torch.Tensor):
        numpy_data = tensor_data.numpy()
        data_to_save = numpy_data.tolist()  
    else:
        data_to_save = tensor_data  

    with open(json_path, 'w') as json_file:
        json.dump(data_to_save, json_file)

    print(f"Successfully converted {pt_path} to {json_path}")

except Exception as e:
    print(f"Error processing {pt_path}: {str(e)}")