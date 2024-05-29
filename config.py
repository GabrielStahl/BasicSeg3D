import torch.nn as nn
import torch.optim as optim
import os

# Data configuration

environment = os.environ.get('ENVIRONMENT', 'local')  # Default to 'local' if the environment variable is not set

if environment == 'local':
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/"
elif environment == 'cluster':
    data_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"


# Model configuration
in_channels = 1
out_channels = 4

# Training configuration
epochs = 2
batch_size = 1
learning_rate = 0.01

# Output configuration
model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/checkpoints/unet_model.pth"