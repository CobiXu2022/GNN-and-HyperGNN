import torch

model_path = 'runs/UniGAT_2_coauthorship_dblp/seed_1/UniGAT_dblp_model.pth'  
model_state_dict = torch.load(model_path, map_location='cpu')


print(model_state_dict)