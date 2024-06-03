import torch
import nibabel as nib
import numpy as np
from model import UNet
import config
from utils import visualize_images
import os
import torch.nn as nn


def preprocess_image(image_path):
    # Load the NIfTI image
    image = nib.load(image_path).get_fdata()
    
    # Normalize the image
    max_value = np.max(image)
    normalized_image = image / max_value
    
    # Add batch dimension and convert to tensor
    input_tensor = torch.from_numpy(normalized_image[np.newaxis, np.newaxis, ...]).float()
    
    return input_tensor

def postprocess_output(output):
    # Convert output tensor to numpy array
    output_numpy = output.detach().cpu().numpy()
    
    # Remove batch dimension and channel dimension
    output_numpy = np.squeeze(output_numpy)
    
    # Apply threshold to obtain binary segmentation mask
    threshold = 0.5
    segmentation_mask = (output_numpy > threshold).astype(np.uint8)
    
    return segmentation_mask

def perform_inference(model, image_path, device):
    # Preprocess the input image
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
    
    # Postprocess the output
    segmentation_mask = postprocess_output(output)
    
    return segmentation_mask

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)

    # To use multiple GPUs, parallelize the model
    model = nn.DataParallel(model)
    
    model.to(device)
    
    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path))
        print(f"Loaded trained model weights from: {config.model_save_path}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path}")
        return
    
    # Set the model to evaluation mode
    model.eval()
    
    # Specify the path to the new, unseen MRI image
    image_path = "path/to/new/mri/image.nii.gz"
    
    # Perform inference
    segmentation_mask = perform_inference(model, image_path, device)
    
    # Visualize the input image and segmentation mask
    input_image = nib.load(image_path).get_fdata()
    visualize_images([input_image, segmentation_mask], ["Input Image", "Segmentation Mask"])
    
    # Save the segmentation mask as a NIfTI file
    output_path = "path/to/save/segmentation_mask.nii.gz"
    segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
    nib.save(segmentation_nifti, output_path)
    print(f"Segmentation mask saved at: {output_path}")

if __name__ == "__main__":
    main()