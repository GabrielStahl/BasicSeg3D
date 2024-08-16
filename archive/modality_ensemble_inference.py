import os
import numpy as np
import nibabel as nib
import torch
from model import UNet
import config
import torch.nn as nn
from data_loader import MRIDataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

class Inference:
    def __init__(self, models):
        self.models = models
        self.crop_size = config.crop_size
        self.original_shape = (240, 240, 155)
        
    def postprocess_output(self, output):
        output_numpy = output.detach().cpu().numpy()
        output_numpy = output_numpy[0, :, :, :]
        intensity_to_class = {
            0: 0, # Background
            1: 2, # Outer tumor region
            2: 4, # Enhancing tumor
            3: 1  # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        segmentation_mask = map_func(output_numpy).astype(np.uint8)
        padded_mask = self.pad_to_original_shape(segmentation_mask, dtype=np.uint8)
        return padded_mask
    
    def pad_to_original_shape(self, mask, dtype=np.uint8):
        depth, height, width = self.original_shape
        crop_depth, crop_height, crop_width = self.crop_size

        pad_depth = (depth - crop_depth) // 2
        pad_height = (height - crop_height) // 2
        pad_width = (width - crop_width) // 2

        padded_mask = np.zeros(self.original_shape, dtype=dtype)
        padded_mask[pad_depth:pad_depth+crop_depth, pad_height:pad_height+crop_height, pad_width:pad_width+crop_width] = mask

        return padded_mask
    
    def inference_modality_ensemble(self, input_data_list, device):
        for model in self.models:
            model.eval()

        ensemble_outputs = []

        with torch.no_grad():
            for model, input_data in zip(self.models, input_data_list):
                input_tensor = input_data[0].to(device)
                output = model(input_tensor)
                output = output.squeeze(0)  # Remove batch dimension
                ensemble_outputs.append(output)

            # Stack outputs from all models
            ensemble_outputs = torch.stack(ensemble_outputs)  # ensemble_outputs.shape: [num_models, classes, d, h, w]
            
            # Apply softmax to convert logits to probabilities
            softmax_probs = F.softmax(ensemble_outputs, dim=1)  # softmax_probs.shape: [num_models, classes, d, h, w]
            
            # Compute mean prediction
            mean_output = torch.mean(softmax_probs, dim=0)  # Shape: [classes, d, h, w]
            
            # Compute segmentation mask
            segmentation_mask = torch.argmax(mean_output, dim=0)  # Shape: [d, h, w]
            segmentation_mask = self.postprocess_output(segmentation_mask.unsqueeze(0))  # Shape: [240, 240, 155]

            # Consider the disagreement between models, compute the entropy of the mean output
            epsilon = 1e-7
            uncertainty = -torch.sum(mean_output * torch.log(mean_output + epsilon), dim=0)
            # normalize the entropy 
            n = mean_output.shape[0]  # number of classes
            uncertainty = uncertainty / torch.log(torch.tensor(n, dtype=torch.float))
            
            # Pad the uncertainty map to the original shape
            uncertainty = self.pad_to_original_shape(uncertainty.cpu().numpy(), dtype=np.float32)

        return segmentation_mask, uncertainty

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create datasets for each modality
    dataset_flair = MRIDataset(config.test_dir, modality="FLAIR_bias")
    dataset_t1c = MRIDataset(config.test_dir, modality="T1c_bias")
    dataset_dti = MRIDataset(config.test_dir, modality="DTI_eddy_FA")

    # Create data loaders
    data_loader_flair = DataLoader(dataset_flair, batch_size=1, shuffle=False)
    data_loader_t1c = DataLoader(dataset_t1c, batch_size=1, shuffle=False)
    data_loader_dti = DataLoader(dataset_dti, batch_size=1, shuffle=False)

    # Load models
    models = []
    modalities = ["FLAIR_bias", "T1c_bias", "DTI_eddy_FA"]
    
    for modality in modalities:
        model = UNet(in_channels=config.in_channels, out_channels=config.out_channels, dropout=config.dropout)
        model.to(device)
        weight_path = os.path.join(config.ensemble_path, f"{modality}_model_0.pth")
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"Loaded trained model weights from: {weight_path}")
            models.append(model)
        except Exception as e:
            print(f"Error loading weights from {weight_path}: {str(e)}")

    inference = Inference(models)

    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Get patient folders
    inference_folders = [f for f in os.listdir(config.test_dir) if f.startswith("UCSF-PDGM")]

    # Perform inference
    for i, (flair_data, t1c_data, dti_data) in enumerate(tqdm(zip(data_loader_flair, data_loader_t1c, data_loader_dti), total=len(inference_folders))):
        patient_folder = inference_folders[i]
        patient_number = patient_folder.split("_")[0].split("-")[-1]
        
        input_data_list = [flair_data, t1c_data, dti_data]
        segmentation_mask, uncertainty_map = inference.inference_modality_ensemble(input_data_list, device)
        
        output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
        nib.save(segmentation_nifti, output_path)
        print(f"Segmentation mask saved at: {output_path}")
        
        uncertainty_path = os.path.join(config.output_dir, f"modality_ensemble_UMap_UCSF-PDGM-{patient_number}.nii.gz")
        uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
        nib.save(uncertainty_nifti, uncertainty_path)
        print(f"Uncertainty map saved at: {uncertainty_path}")

if __name__ == "__main__":
    main()