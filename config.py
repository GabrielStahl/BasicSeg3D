import torch.nn as nn
import torch.optim as optim
import os

# Data configuration

environment = os.environ.get('ENVIRONMENT', 'local')  # Default to 'local' if the environment variable is not set

if environment == 'local':
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/" 
    train_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/"
    val_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/val_data/"
    test_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/test_data/"
    model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/checkpoints/"
    ensemble_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/checkpoints/modality_ensemble/"
    output_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/Predicted_Segmentations/"
    print('Environment is: local')
elif environment == 'cluster':
    data_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
    train_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/TRAIN_SET/"
    val_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/VAL_SET/"
    test_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/TEST_SET/"
    model_save_path = '/cluster/project2/UCSF_PDGM_dataset/BasicSeg/Checkpoints/'
    ensemble_path = "/cluster/project2/UCSF_PDGM_dataset/BasicSeg/Checkpoints/modality_ensemble/"
    output_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/predictions_train_set/" # CHOOSE FROM: predictions_train_set, predictions_test_set
    print('Environment is: cluster')


# Model configuration
in_channels = 1
out_channels = 4
dropout = 0.3

# Data configuration
crop_size = (150, 180, 155)

# Training configuration
if environment == 'local':
    batch_size = 1
    learning_rate = 0.01
    epochs = 2
elif environment == 'cluster':
    batch_size = 1
    learning_rate = 0.001
    epochs = 100

# Uncertainty quantification configuration
uncertainty_method = "modality_ensemble"  # Options: "none", "softmax", "deep_ensemble", "test_time_augmentation", "dropout", "modality_ensemble"

if uncertainty_method == "dropout":
    dropout = 0.5