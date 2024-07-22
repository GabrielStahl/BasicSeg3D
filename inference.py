import os
import numpy as np
import nibabel as nib
import torch
from model import UNet
import config
import torch.nn as nn
from data_loader import MRIDataset
import torchvision.transforms as v2
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
# import torchio as tio # Uncomment this line after installing torchio on the cluster

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
        self.crop_size = config.crop_size
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
        padded_mask = self.pad_to_original_shape(segmentation_mask, dtype=np.uint8)
        
        return padded_mask
    
    def pad_to_original_shape(self, mask, dtype=np.uint8):
        """ Pad the segmentation mask to the original shape
        
        Args: 
            mask (np.ndarray): The mask to pad
            dtype: The data type of the output mask
        """

        depth, height, width = self.original_shape
        crop_depth, crop_height, crop_width = self.crop_size

        pad_depth = (depth - crop_depth) // 2
        pad_height = (height - crop_height) // 2
        pad_width = (width - crop_width) // 2

        padded_mask = np.zeros(self.original_shape, dtype=dtype)
        padded_mask[pad_depth:pad_depth+crop_depth, pad_height:pad_height+crop_height, pad_width:pad_width+crop_width] = mask

        return padded_mask

    
    def perform_inference_none(self, data_loader, device):
        """ Perform inference without uncertainty estimation"""
        self.model.eval()
        segmentation_masks = []

        with torch.no_grad():
            for input_tensor, _ in tqdm(data_loader):
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
            for input_tensor, _ in tqdm(data_loader):
                input_tensor = input_tensor.to(device)
                output = self.model(input_tensor)
                
                # Apply softmax to obtain class probabilities
                output_probabilities = nn.functional.softmax(output, dim=1)
                
                # Get the uncertainty map
                uncertainty_map = 1 - torch.max(output_probabilities, dim=1)[0]

                # squeeze batch dimension, pad to original shape
                uncertainty_map = uncertainty_map.squeeze(0)
                uncertainty_map = self.pad_to_original_shape(uncertainty_map.detach().cpu().numpy(), dtype=np.float32)

                # Apply argmax to obtain the class indices
                output = torch.argmax(output_probabilities, dim=1)
                
                # Postprocess the output
                segmentation_mask = self.postprocess_output(output)
                
                segmentation_masks.append(segmentation_mask)
                uncertainties.append(uncertainty_map)
        
        return segmentation_masks, uncertainties
    
    def perform_inference_test_time_augmentation(self, test_loader, device, augmentationRounds=2):
        self.model.eval()
        
        # Define the test-time augmentation transforms
        test_transforms = tio.Compose([
            tio.RandomAffine(scales=(1, 1), translation=(0.05, 0.05, 0.05)),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        ])
        
        segmentation_masks = []
        uncertainties = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                batch_predictions = []
                for _ in range(augmentationRounds):
                    if len(images.shape) > 4:
                        images = images.squeeze(0)
                    
                    # Create a TorchIO subject which can store transform history for later reversal
                    subject = tio.Subject({"image": tio.ScalarImage(tensor=images)})
                    
                    # Apply transform, send to device, and get model predictions
                    transformed = test_transforms(subject)
                    augmented_images = transformed['image'].data.unsqueeze(0).to(device)
                    classifications_logits = self.model(augmented_images)
                    
                    # Apply inverse transform to predictions to get back to original orientation
                    inverse_transform = transformed.get_inverse_transform()
                    classifications_logits = classifications_logits.squeeze(0)
                    classifications_logits_native = inverse_transform(classifications_logits.cpu())
                    
                    outputs = classifications_logits_native.squeeze()
                    batch_predictions.append(outputs.numpy())
                
                # Convert batch_predictions to labels
                batch_predictions = np.array(batch_predictions)
                batch_labels = np.argmax(batch_predictions, axis=1)
                
                # Perform majority voting
                segmentation_mask_cropped = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=batch_labels)
                
                # Pad the segmentation mask to the original shape
                segmentation_mask = self.pad_to_original_shape(segmentation_mask_cropped, dtype=np.uint8)
                segmentation_masks.append(segmentation_mask)
                
                # Compute entropy as uncertainty measure
                epsilon = 1e-8
                softmax_probs = torch.softmax(torch.tensor(batch_predictions), dim=1)
                softmax_probs = softmax_probs + epsilon
                entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=0)
                entropy = entropy.cpu().numpy()
                
                # Compute uncertainty as the mean entropy across all classes
                uncertainty = np.mean(entropy, axis=0)
                
                # Pad the uncertainty map to the original shape
                uncertainty = self.pad_to_original_shape(uncertainty, dtype=np.float32)
                uncertainties.append(uncertainty)
        
        return segmentation_masks, uncertainties

    def perform_inference_dropout(self, data_loader, device, num_iterations=10):
        """ Perform inference with dropout for uncertainty estimation
        
        Args:
            data_loader (torch.utils.data.DataLoader): The data loader
            device (torch.device): The device to run inference on
            num_iterations (int): The number of iterations to perform for dropout uncertainty estimation
        
        Returns:
            segmentation_masks (list): List of segmentation masks, each of shape (240, 240, 155)
            uncertainties (list): List of uncertainty maps, each of shape (240, 240, 155)
        """


        self.model.train()  # Set the model to train mode to enable dropout
        segmentation_masks = []
        uncertainties = []

        with torch.no_grad():
            for input_tensor, _ in tqdm(data_loader):
                input_tensor = input_tensor.to(device)
                outputs = []

                for _ in range(num_iterations):
                    output = self.model(input_tensor)
                    output = nn.functional.softmax(output, dim=1)
                    outputs.append(output)

                # Compute the mean and variance of the outputs
                outputs = torch.stack(outputs)
                mean_output = torch.mean(outputs, dim=0)
                var_output = torch.var(outputs, dim=0)

                # Apply argmax to obtain the class indices
                segmentation_mask = torch.argmax(mean_output, dim=1)
                segmentation_mask = self.postprocess_output(segmentation_mask)
                segmentation_masks.append(segmentation_mask)

                # Compute the epistemic uncertainty
                uncertainty = torch.mean(var_output, dim=1)
                uncertainty = self.pad_to_original_shape(uncertainty.squeeze().cpu().numpy(), dtype=np.float32)
                uncertainties.append(uncertainty)

        return segmentation_masks, uncertainties
    
    def perform_inference_dropout_single(self, input_data, device, num_iterations=10):
        self.model.train()  # Set the model to train mode to enable dropout
        
        # input_data is a tuple (input_tensor, target_tensor)
        input_tensor = input_data[0].to(device)
        
        outputs = []

        with torch.no_grad():
            for _ in range(num_iterations):
                output = self.model(input_tensor)
                output = nn.functional.softmax(output, dim=1)
                outputs.append(output)

        # Compute the mean and variance of the outputs
        outputs = torch.stack(outputs)
        mean_output = torch.mean(outputs, dim=0)
        var_output = torch.var(outputs, dim=0)

        # Apply argmax to obtain the class indices
        segmentation_mask = torch.argmax(mean_output, dim=1)
        segmentation_mask = self.postprocess_output(segmentation_mask)

        # Compute the epistemic uncertainty
        uncertainty = torch.mean(var_output, dim=1)
        uncertainty = self.pad_to_original_shape(uncertainty.squeeze().cpu().numpy(), dtype=np.float32)

        return segmentation_mask, uncertainty
    
    def perform_inference_deep_ensemble(self, data_loader, device, models):
        for model in models:
            model.eval()

        segmentation_masks = []
        uncertainties = []

        with torch.no_grad():
            for input_tensor, _ in tqdm(data_loader):
                input_tensor = input_tensor.to(device)
                ensemble_outputs = []

                for model in models:
                    output = model(input_tensor)
                    output = output.squeeze(0) # Remove batch dimension
                    ensemble_outputs.append(output)

                # Stack outputs from all models
                ensemble_outputs = torch.stack(ensemble_outputs)  # Shape: [num_models, classes, d, h, w]
                
                # Apply softmax to convert logits to probabilities
                softmax_probs = F.softmax(ensemble_outputs, dim=1) # Shape: [num_models, classes, d, h, w]
                
                # Compute mean prediction
                mean_output = torch.mean(softmax_probs, dim=0)  # Shape: [classes, d, h, w]
                
                # Compute segmentation mask
                segmentation_mask = torch.argmax(mean_output, dim=0)  # Shape: [d, h, w]
                segmentation_mask = self.postprocess_output(segmentation_mask.unsqueeze(0))  # Shape: [240, 240, 155]
                segmentation_masks.append(segmentation_mask)

                # Compute entropy as uncertainty measure
                epsilon = 1e-8
                entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + epsilon), dim=1)  # Shape: [num_models, classes, d, h, w]
                
                # Compute uncertainty as the mean entropy across all models
                uncertainty = torch.mean(entropy, dim=0)  # Shape: [batch, d, h, w]
                
                # Pad the uncertainty map to the original shape
                uncertainty = self.pad_to_original_shape(uncertainty.cpu().numpy(), dtype=np.float32)
                uncertainties.append(uncertainty)

        return segmentation_masks, uncertainties

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels, dropout=config.dropout)
    model.to(device)

    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        weights = "new_model_1_best_epoch.pth"
        model_save_path = os.path.join(config.model_save_path, weights)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded trained model weights from: {config.model_save_path + weights}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path + weights}")
        return

    # Set the model to evaluation mode
    model.eval()

    # Split the data into train, validation, and test sets
    inference_folders, _, _ = MRIDataset.split_data(config.data_dir, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, seed=42)

    print(f"Getting patients from directory: {config.data_dir}")
    print(f"Performing inference on: {len(inference_folders)} patients")

    # Create an instance of the Inference class based on the selected uncertainty estimation method
    inference = Inference(model, config.uncertainty_method)
    
    dataset = MRIDataset(config.data_dir, inference_folders)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Perform inference on each validation patient
    if config.uncertainty_method == "none":
        segmentation_masks = inference.perform_inference_none(data_loader, device)
        
        for i, segmentation_mask in enumerate(segmentation_masks):
            patient_number = inference_folders[i].split("_")[0].split("-")[-1]
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
    
    elif config.uncertainty_method == "softmax":
        segmentation_masks, uncertainties = inference.perform_inference_softmax(data_loader, device)
        
        for i, (segmentation_mask, uncertainty_map) in enumerate(zip(segmentation_masks, uncertainties)):
            patient_number = inference_folders[i].split("_")[0].split("-")[-1]
            
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
            patient_number = inference_folders[i].split("_")[0].split("-")[-1]
            
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

    elif config.uncertainty_method == "dropout":
        for i, input_data in enumerate(tqdm(data_loader)):
            patient_number = inference_folders[i].split("_")[0].split("-")[-1]
            
            # Perform inference for a single patient
            segmentation_mask, uncertainty_map = inference.perform_inference_dropout_single(input_data, device)
            
            # Save segmentation mask
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            # Save uncertainty map
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

    elif config.uncertainty_method == "deep_ensemble":
        models = []
        for i in range(3):  # Assuming we have 3 models in the ensemble
            model = UNet(in_channels=config.in_channels, out_channels=config.out_channels, dropout=config.dropout)
            model.to(device)
            
            # Load the trained model weights
            weights = f"new_model_{i}_best_epoch.pth"
            model_save_path = os.path.join(config.model_save_path, weights)
            if os.path.exists(model_save_path):
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                print(f"Loaded trained model weights from: {model_save_path}")
            else:
                print(f"Trained model weights not found at: {model_save_path}")
                return
        
        models.append(model)

        inference = Inference(models[0], config.uncertainty_method)  # We pass the first model, but it won't be used
        segmentation_masks, uncertainties = inference.perform_inference_deep_ensemble(data_loader, device, models)
        
        for i, (segmentation_mask, uncertainty_map) in enumerate(zip(segmentation_masks, uncertainties)):
            patient_number = inference_folders[i].split("_")[0].split("-")[-1]
            
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

if __name__ == "__main__":
    main()