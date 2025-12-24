import json

with open('/home/zmxu/Desktop/ly/UniGNN/runs/UniGAT_2_cocitation_pubmed/seed_1/features_epoch_0_UniGAT_pubmed.json', 'r') as json_file:
    data = json.load(json_file)
print(f"Data type: {type(data)}")
print(f"Number of entries: {len(data)}")

if len(data) > 0:
    print(f"Number of elements in the first entry: {len(data[0])}")