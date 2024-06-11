import os
import numpy as np
import nibabel as nib
import torch
from model import UNet
import config
import torch.nn as nn
from data_loader import MRIDataset
import torchvision.transforms as v2
from torchvision.transforms import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchio as tio

class Inference:
    """ Perform inference using the trained model, optionally with uncertainty maps"""

    def __init__(self, model, uncertainty_method):
        """ Initialize the Inference class
        Args:
            model (torch.nn.Module): The trained model
            uncertainty_method (str): The method to estimate uncertainty            
        """

        self.model = model
        self.uncertainty_method = uncertainty_method
        self.crop_size = (150, 180, 116)
        self.original_shape = (240, 240, 155)

        # Define transforms for test-time augmentation 
        self.test_transforms = v2.Compose([
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        
    def postprocess_output(self, output):
        """ Postprocess the model output
        --> Remove batch dimension, map class indices to original class values, pad to original shape

        Args:
            output (torch.Tensor): The model output tensor

        Returns:
            np.ndarray: The postprocessed segmentation mask
        
        """

        # Convert output tensor to numpy array
        output_numpy = output.detach().cpu().numpy()
        
        # Remove batch dimension
        output_numpy = output_numpy[0, :, :, :]
        
        # Map the class indices to the original intensity values
        intensity_to_class = {
            0: 0, # Background
            1: 2, # Outer tumor region
            2: 4, # Enhancing tumor
            3: 1  # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        segmentation_mask = map_func(output_numpy).astype(np.uint8)
        
        # Pad the segmentation mask to the original shape
        padded_mask = self.pad_to_original_shape(segmentation_mask)
        
        return padded_mask
    
    def pad_to_original_shape(self, segmentation_mask):
        """ Pad the segmentation mask to the original shape
        
        Args: 
            segmentation_mask (np.ndarray): The segmentation mask to pad
        """

        depth, height, width = self.original_shape
        crop_depth, crop_height, crop_width = self.crop_size

        pad_depth = (depth - crop_depth) // 2
        pad_height = (height - crop_height) // 2
        pad_width = (width - crop_width) // 2

        padded_mask = np.zeros(self.original_shape, dtype=np.uint8)
        padded_mask[pad_depth:pad_depth+crop_depth, pad_height:pad_height+crop_height, pad_width:pad_width+crop_width] = segmentation_mask

        return padded_mask
    
    def perform_inference_none(self, data_loader, device):
        """ Perform inference without uncertainty estimation"""
        self.model.eval()
        segmentation_masks = []

        with torch.no_grad():
            for input_tensor, _ in data_loader:
                input_tensor = input_tensor.to(device)
                output = self.model(input_tensor)
                
                # Apply softmax to obtain class probabilities
                output = nn.functional.softmax(output, dim=1)
                
                # Apply argmax to obtain the class indices
                output = torch.argmax(output, dim=1)
                
                # Postprocess the output
                segmentation_mask = self.postprocess_output(output)
                segmentation_masks.append(segmentation_mask)
        
        return segmentation_masks
    
    def perform_inference_softmax(self, data_loader, device):
        """ Perform inference with softmax uncertainty estimation"""
        self.model.eval()
        segmentation_masks = []
        uncertainties = []

        with torch.no_grad():
            for input_tensor, _ in data_loader:
                input_tensor = input_tensor.to(device)
                output = self.model(input_tensor)
                
                # Apply softmax to obtain class probabilities
                output_probabilities = nn.functional.softmax(output, dim=1)
                
                # Get the uncertainty map
                uncertainty_map = torch.max(output_probabilities, dim=1)[0]

                # squeeze batch dimension, pad to original shape
                uncertainty_map = uncertainty_map.squeeze(0)
                uncertainty_map = self.pad_to_original_shape(uncertainty_map.detach().cpu().numpy())
                
                # Apply argmax to obtain the class indices
                output = torch.argmax(output_probabilities, dim=1)
                
                # Postprocess the output
                segmentation_mask = self.postprocess_output(output)
                
                segmentation_masks.append(segmentation_mask)
                uncertainties.append(uncertainty_map)
        
        return segmentation_masks, uncertainties

    def perform_inference_test_time_augmentation_old(self, data_loader, device, augmentation_rounds=2):
        """ Perform inference with test-time augmentation for aleatoric uncertainty estimation"""
        self.model.eval()
        segmentation_masks = []
        uncertainties = []

        with torch.no_grad():
            for r in range(augmentation_rounds):
                print(f"Augmentation round: {r+1}/{augmentation_rounds}")

                batch_predictions = []
                for input_tensor, _ in data_loader:
                    input_tensor = input_tensor.to(device)
                    output = self.model(input_tensor)
                    batch_predictions.append(output.cpu().numpy())

                # Convert batch_predictions to labels
                batch_predictions = np.array(batch_predictions)  # shape: (batch_size, num_classes, depth, height, width)
                batch_labels = np.argmax(batch_predictions, axis=1)  # shape: (batch_size, depth, height, width)

                # Perform majority voting
                majority_labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=batch_labels)

                # Compute entropy as uncertainty measure
                softmax_probs = torch.softmax(torch.tensor(batch_predictions), dim=1)
                entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=1)
                entropy = entropy.cpu().numpy()

                # Postprocess the output
                segmentation_mask = self.postprocess_output(torch.tensor(majority_labels))

                segmentation_masks.append(segmentation_mask)
                uncertainties.append(np.mean(entropy, axis=0))

        return segmentation_masks, uncertainties
    
    def perform_inference_test_time_augmentation(self, test_loader, device, augmentationRounds=2):
        """
        Define custom test step function that uses test-time augmentation to estimate aleatoric uncertainty

        Args:
            model (torch model): Model to be evaluated
            test_loader (custom DataLoader): DataLoader for test set
            augmentationRounds (int): Number of augmentations to perform
        
        """
        self.model.eval()

        # Define the test-time augmentation transforms
        # ATTENTION: LESS IS MORE here. Too many augmentations lead to decrease in model performance
        test_transforms = tio.Compose([
            tio.RandomAffine(scales=(1, 1), translation=(0.05, 0.05, 0.05)),  # Random affine transformation
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Random flip
        ])

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                batch_predictions = []
                for _ in range(augmentationRounds):

                    # only squeeze if dim > 4
                    if len(images.shape) > 4:
                        images = images.squeeze(0)
                    augmented_images = test_transforms(images)                    
                    augmented_images = augmented_images.unsqueeze(0).to(device)
                    classifications_logits = self.model(augmented_images)
                    outputs = classifications_logits[0] # store first logits as 'outputs' to maintain convention                
                    batch_predictions.append(outputs.cpu().numpy())

                # Convert batch_predictions to labels
                batch_predictions = np.array(batch_predictions)  # shape: (augmentationRounds, batch_size, num_classes)
                batch_labels = np.argmax(batch_predictions, axis=1)  # shape: (augmentationRounds, batch_size)

                # Perform majority voting
                segmentation_masks = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=batch_labels)

                labels = labels.float().to(device)

                # Compute entropy as uncertainty measure
                softmax_probs = torch.softmax(torch.tensor(batch_predictions), dim=1)
                entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=0)
                entropy = entropy.cpu().numpy()
                uncertainty = np.max(entropy, axis=0)

        return segmentation_masks, uncertainty


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model.to(device)

    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        model_save_path = os.path.join(config.model_save_path, "epoch_20_cluster.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded trained model weights from: {config.model_save_path}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path}")
        return

    # Set the model to evaluation mode
    model.eval()

    # Split the data into train, validation, and test sets
    train_folders, val_folders, _ = MRIDataset.split_data(config.data_dir, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    print(f"Number of validation patients: {len(train_folders)}")

    # Create an instance of the Inference class based on the selected uncertainty estimation method
    inference = Inference(model, config.uncertainty_method)
    
    dataset = MRIDataset(config.data_dir, train_folders)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Perform inference on each validation patient
    if config.uncertainty_method == "none":
        segmentation_masks = inference.perform_inference_none(data_loader, device)
        
        for i, segmentation_mask in enumerate(segmentation_masks):
            patient_number = train_folders[i].split("_")[0].split("-")[-1]
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
    
    elif config.uncertainty_method == "softmax":
        segmentation_masks, uncertainties = inference.perform_inference_softmax(data_loader, device)
        
        for i, (segmentation_mask, uncertainty_map) in enumerate(zip(segmentation_masks, uncertainties)):
            patient_number = train_folders[i].split("_")[0].split("-")[-1]
            
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

    elif config.uncertainty_method == "test_time_augmentation":
        segmentation_masks, uncertainties = inference.perform_inference_test_time_augmentation(data_loader, device)
        
        for i, (segmentation_mask, uncertainty_map) in enumerate(zip(segmentation_masks, uncertainties)):
            patient_number = train_folders[i].split("_")[0].split("-")[-1]
            
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_mask = segmentation_mask.astype(np.int32) # Convert segmentation_mask to int32 or float32

            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_map = uncertainty_map.astype(np.float32) # Convert uncertainty_map to int32 or float32
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

if __name__ == "__main__":
    main()