import torch.nn as nn
import torch.optim as optim

# Data configuration
data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/"

# Model configuration
in_channels = 1
out_channels = 3

# Training configuration
epochs = 2
batch_size = 2
learning_rate = 0.01

# Output configuration
model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/checkpoints/unet_model.pth"