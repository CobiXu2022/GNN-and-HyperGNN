import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

# Step 1: Load the features and labels
features = torch.load('/home/zmxu/Desktop/ly/UniGNN/runs/UniGCNII_32_cocitation_pubmed/seed_1/features_epoch_200_UniGCNII_pubmed.pt', map_location=torch.device('cpu'))
labels = torch.load('saved_labels_pubmed.pt', map_location=torch.device('cpu'))

# Convert features to tensor if it's a list
if isinstance(features, list):
    features = torch.stack(features)

# Print shapes
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Ensure matching lengths
assert features.shape[0] == labels.shape[0], "Features and labels must have the same length."

# Convert to NumPy and handle NaNs
features = features.detach().cpu().numpy()
labels = labels.numpy()
features = np.nan_to_num(features, nan=0.0)

# Step 2: Apply UMAP
#reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
reducer = umap.UMAP()
embedding = reducer.fit_transform(features)

# Step 3: Plot with legend
plt.figure(figsize=(12, 8))  # Slightly wider to accommodate legend

# Get unique labels and assign colors
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))  # Use Spectral colormap

# Plot each label group with a distinct color and label
for label, color in zip(unique_labels, colors):
    mask = labels == label
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                c=[color],  # Use list to ensure consistent color
                label=str(label),  # Convert label to string for legend
                s=5)

# Add legend outside the plot
plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save and show
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('pubmed_0.png', bbox_inches='tight')  # bbox_inches prevents legend cutoff
plt.show()