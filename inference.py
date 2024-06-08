import os
import numpy as np
import nibabel as nib
import torch
from model import UNet
import config
from utils import visualize_images
import torch.nn as nn
from data_loader import MRIDataset

def preprocess_image(image_path, crop_size=(150, 180, 116)):
    # Load the NIfTI image
    image = nib.load(image_path).get_fdata()
    
    # Center crop the input image
    image = center_crop(image, crop_size)
    
    # Normalize the image
    max_value = np.max(image)
    normalized_image = image / max_value if max_value > 0 else image
    
    # Add batch dimension and convert to tensor
    input_tensor = torch.from_numpy(normalized_image[np.newaxis, np.newaxis, ...]).float()
    
    return input_tensor

def center_crop(image, crop_size):
    depth, height, width = image.shape
    crop_depth, crop_height, crop_width = crop_size

    start_depth = (depth - crop_depth) // 2
    start_height = (height - crop_height) // 2
    start_width = (width - crop_width) // 2

    cropped_image = image[start_depth:start_depth+crop_depth, start_height:start_height+crop_height, start_width:start_width+crop_width]
    return cropped_image

def postprocess_output(output, original_shape=(240, 240, 155), crop_size=(150, 180, 116)):
    # Convert output tensor to numpy array
    output_numpy = output.detach().cpu().numpy()
    
    # Apply argmax to obtain the class indices
    output_numpy = np.argmax(output_numpy, axis=1)
    
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
    padded_mask = pad_to_original_shape(segmentation_mask, original_shape, crop_size)
    
    return padded_mask

def pad_to_original_shape(segmentation_mask, original_shape, crop_size):
    depth, height, width = original_shape
    crop_depth, crop_height, crop_width = crop_size

    pad_depth = (depth - crop_depth) // 2
    pad_height = (height - crop_height) // 2
    pad_width = (width - crop_width) // 2

    padded_mask = np.zeros(original_shape, dtype=np.uint8)
    padded_mask[pad_depth:pad_depth+crop_depth, pad_height:pad_height+crop_height, pad_width:pad_width+crop_width] = segmentation_mask

    return padded_mask

def perform_inference(model, image_path, device):
    crop_size = (150, 180, 116)
    
    # Preprocess the input image
    input_tensor = preprocess_image(image_path, crop_size)
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess the output
    segmentation_mask = postprocess_output(output, original_shape=(240, 240, 155), crop_size=crop_size)
    
    return segmentation_mask

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

    # Perform inference on each validation patient
    for patient_folder in train_folders:
        patient_number = patient_folder.split("_")[0].split("-")[-1]
        image_path = os.path.join(config.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_T2_bias.nii.gz")
        print(f"Performing inference on: {image_path}")
        segmentation_mask = perform_inference(model, image_path, device)

        # Save the segmentation mask as a NIfTI file
        output_path = os.path.join(config.output_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        segmentation_nifti = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))
        nib.save(segmentation_nifti, output_path)
        print(f"Segmentation mask saved at: {output_path}")

if __name__ == "__main__":
    main()
