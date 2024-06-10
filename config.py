import torch.nn as nn
import torch.optim as optim
import os

# Data configuration

environment = os.environ.get('ENVIRONMENT', 'local')  # Default to 'local' if the environment variable is not set

if environment == 'local':
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/"
    model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/checkpoints/"
    output_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/Predicted_Segmentations/"
    print('Environment is: local')
elif environment == 'cluster':
    data_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
    model_save_path = '/cluster/project2/UCSF_PDGM_dataset/BasicSeg/Checkpoints/'
    output_dir = "/cluster/project2/UCSF_PDGM_dataset/BasicSeg/Predicted_Segmentations/"
    print('Environment is: cluster')


# Model configuration
in_channels = 1
out_channels = 4

# Training configuration
epochs = 50
if environment == 'local':
    batch_size = 1
    learning_rate = 0.01
elif environment == 'cluster':
    batch_size = 1
    learning_rate = 0.01

# Uncertainty quantification configuration
uncertainty_method = "softmax"  # Options: "none", "softmax", "deep_ensemble", "test_time_augmentation"