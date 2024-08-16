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

# Define the modalities you want to use in the ensemble
MODALITIES = ["T1c_bias", "DTI_eddy_FA"]  # Add or remove modalities as needed , "T1c_bias", "DTI_eddy_FA"

# Choose which uncertainty map to use: 'probability' or 'disagreement'
UNCERTAINTY_TYPE = 'probability'  # Change this to 'disagreement' if you want to use the disagreement-based uncertainty

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
        class_predictions = []
        
        with torch.no_grad():
            for model, input_data in zip(self.models, input_data_list):
                input_tensor = input_data[0].to(device)
                output = model(input_tensor)
                output = output.squeeze(0)  # Remove batch dimension
                ensemble_outputs.append(output)
                class_predictions.append(torch.argmax(output, dim=0))  # Get class predictions

        # Stack outputs from all models
        ensemble_outputs = torch.stack(ensemble_outputs)  # Shape: [num_models, classes, d, h, w]
        
        # Apply softmax to convert logits to probabilities
        softmax_probs = F.softmax(ensemble_outputs, dim=1)  # Shape: [num_models, classes, d, h, w]
        
        # Compute mean prediction
        mean_output = torch.mean(softmax_probs, dim=0)  # Shape: [classes, d, h, w]
        
        # Compute segmentation mask
        segmentation_mask = torch.argmax(mean_output, dim=0)  # Shape: [d, h, w]
        segmentation_mask = self.postprocess_output(segmentation_mask.unsqueeze(0))  # Shape: [240, 240, 155]
        
        # Calculate normalized entropy of mean output
        epsilon = 1e-7
        n_classes = mean_output.shape[0]
        uncertainty_prob = -torch.sum(mean_output * torch.log(mean_output + epsilon), dim=0)
        uncertainty_prob = uncertainty_prob / torch.log(torch.tensor(n_classes, dtype=torch.float))
        
        # Calculate entropy based on model disagreement
        stacked_predictions = torch.stack(class_predictions)  # Shape: [num_models, d, h, w]
        class_freq = torch.zeros((n_classes, *stacked_predictions.shape[1:]), device=device)
        for i in range(n_classes):
            class_freq[i] = (stacked_predictions == i).float().mean(dim=0)
        
        uncertainty_disagreement = -torch.sum(class_freq * torch.log(class_freq + epsilon), dim=0)
        uncertainty_disagreement = uncertainty_disagreement / torch.log(torch.tensor(n_classes, dtype=torch.float))
        
        # Pad both uncertainty maps to the original shape
        uncertainty_prob = self.pad_to_original_shape(uncertainty_prob.cpu().numpy(), dtype=np.float32)
        uncertainty_disagreement = self.pad_to_original_shape(uncertainty_disagreement.cpu().numpy(), dtype=np.float32)
        
        return segmentation_mask, uncertainty_prob, uncertainty_disagreement

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create datasets and data loaders for each modality
    datasets = {modality: MRIDataset(config.test_dir, modality=modality) for modality in MODALITIES}
    data_loaders = {modality: DataLoader(dataset, batch_size=1, shuffle=False) for modality, dataset in datasets.items()}

    # Load models
    models = []
    
    for modality in MODALITIES:
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
    for i, data_batch in enumerate(tqdm(zip(*data_loaders.values()), total=len(inference_folders))):
        patient_folder = inference_folders[i]
        patient_number = patient_folder.split("_")[0].split("-")[-1]
        
        input_data_list = list(data_batch)
        segmentation_mask, uncertainty_prop, uncertainty_class = inference.inference_modality_ensemble(input_data_list, device)

        # conditionally save the uncertainty map based on the chosen uncertainty type
        if UNCERTAINTY_TYPE == 'probability':
            uncertainty_map = uncertainty_prop
        elif UNCERTAINTY_TYPE == 'disagreement':
            uncertainty_map = uncertainty_class
        
        output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
        nib.save(segmentation_nifti, output_path)
        print(f"Segmentation mask saved at: {output_path}")
        
        uncertainty_path = os.path.join(config.output_dir, f"modality_ensemble_UMap_{UNCERTAINTY_TYPE}_UCSF-PDGM-{patient_number}.nii.gz")
        uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
        nib.save(uncertainty_nifti, uncertainty_path)
        print(f"Uncertainty map ({UNCERTAINTY_TYPE}-based) saved at: {uncertainty_path}")

if __name__ == "__main__":
    main()