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
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        
    def preprocess_image(self, image_path):
        """ Preprocess the input image
        --> centre crop, normalize, add batch dimension, convert to tensor

        Args:
            image_path (str): The path to the input image
        """

        # Load the NIfTI image
        image = nib.load(image_path).get_fdata()
        
        # Center crop the input image
        image = self.center_crop(image, self.crop_size)
        
        # Normalize the image
        max_value = np.max(image)
        normalized_image = image / max_value if max_value > 0 else image
        
        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(normalized_image[np.newaxis, np.newaxis, ...]).float()
        
        return input_tensor
    
    def center_crop(self, image, crop_size):
        """ Centre crop the input image
        Args:
            image (np.ndarray): The input image
            crop_size (tuple): The size to crop the image to
        
        Returns:
            np.ndarray: The centre cropped image
        """

        depth, height, width = image.shape
        crop_depth, crop_height, crop_width = crop_size

        start_depth = (depth - crop_depth) // 2
        start_height = (height - crop_height) // 2
        start_width = (width - crop_width) // 2

        cropped_image = image[start_depth:start_depth+crop_depth, start_height:start_height+crop_height, start_width:start_width+crop_width]
        return cropped_image
    
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
    
    def perform_inference_none(self, image_path, device):
        """ Perform inference without uncertainty estimation"""

        # Preprocess the input image
        input_tensor = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Apply softmax to obtain class probabilities
        output = nn.functional.softmax(output, dim=1)
        
        # Apply argmax to obtain the class indices
        output = torch.argmax(output, dim=1)
        
        # Postprocess the output
        segmentation_mask = self.postprocess_output(output)
        
        return segmentation_mask
    
    def perform_inference_softmax(self, image_path, device):
        """ Perform inference with softmax uncertainty estimation"""

        # Preprocess the input image
        input_tensor = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # Perform inference
        with torch.no_grad():
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
        
        return segmentation_mask, uncertainty_map

def perform_inference_test_time_augmentation(self, data_loader, device, augmentation_rounds=7):
    self.model.eval()
    segmentation_masks = []
    uncertainties = []

    with torch.no_grad():
        for input_tensor, _ in data_loader:
            input_tensor = input_tensor.to(device)

            # Perform test-time augmentation
            batch_predictions = []
            for _ in range(augmentation_rounds):
                augmented_image = self.test_transforms(input_tensor)
                output = self.model(augmented_image)
                batch_predictions.append(output.cpu().numpy())

            # Convert batch_predictions to labels
            batch_predictions = np.array(batch_predictions)  # shape: (augmentation_rounds, batch_size, num_classes, depth, height, width)
            batch_labels = np.argmax(batch_predictions, axis=2)  # shape: (augmentation_rounds, batch_size, depth, height, width)

            # Perform majority voting
            majority_labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=batch_labels)

            # Compute entropy as uncertainty measure
            softmax_probs = torch.softmax(torch.tensor(batch_predictions), dim=2)
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=2)
            entropy = entropy.cpu().numpy()

            # Postprocess the output
            segmentation_mask = self.postprocess_output(torch.tensor(majority_labels))
            print(f"shape of segmentation mask: {segmentation_mask.shape}")

            segmentation_masks.append(segmentation_mask)
            uncertainties.append(np.mean(entropy, axis=(0, 1)))

    return segmentation_masks, uncertainties

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model.to(device)

    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        model_save_path = os.path.join(config.model_save_path, "final_epoch.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded trained model weights from: {config.model_save_path}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path}")
        return

    # Set the model to evaluation mode
    model.eval()

    # Split the data into train, validation, and test sets
    train_folders, val_folders, _ = MRIDataset.split_data(config.data_dir)

    print(f"Number of validation patients: {len(train_folders)}")
    
    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Create an instance of the Inference class based on the selected uncertainty estimation method
    inference = Inference(model, config.uncertainty_method)

    # Perform inference on each validation patient
    for patient_folder in train_folders:
        patient_number = patient_folder.split("_")[0].split("-")[-1]
        image_path = os.path.join(config.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_T2_bias.nii.gz")
        print(f"Performing inference on: {image_path}")
        
        if config.uncertainty_method == "none":
            segmentation_mask = inference.perform_inference_none(image_path, device)
            
            # Save the segmentation mask as a NIfTI file
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
        
        elif config.uncertainty_method == "softmax":
            segmentation_mask, uncertainty_map = inference.perform_inference_softmax(image_path, device)
            
            # Save the segmentation mask as a NIfTI file
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            # Save the uncertainty map as a NIfTI file
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

        elif config.uncertainty_method == "test_time_augmentation":
            segmentation_mask, uncertainty_map = inference.perform_inference_test_time_augmentation(image_path, device)
            
            # Save the segmentation mask as a NIfTI file
            output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
            segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
            nib.save(segmentation_nifti, output_path)
            print(f"Segmentation mask saved at: {output_path}")
            
            # Save the uncertainty map as a NIfTI file
            uncertainty_path = os.path.join(config.output_dir, f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
            uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=np.eye(4))
            nib.save(uncertainty_nifti, uncertainty_path)
            print(f"Uncertainty map saved at: {uncertainty_path}")

if __name__ == "__main__":
    main()