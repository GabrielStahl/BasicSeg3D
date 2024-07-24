import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import config

class MRIDataset(Dataset):
    def __init__(self, data_dir, modality = "FLAIR_bias", crop_size=config.crop_size, transform=None):
        self.data_dir = data_dir
        self.patient_folders = [folder for folder in os.listdir(data_dir) if folder.startswith("UCSF-PDGM-") and "FU" not in folder]
        self.crop_size = crop_size
        self.transform = transform
        self.modality = modality

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, index):
        patient_folder = self.patient_folders[index]
        patient_number = patient_folder.split("_")[0].split("-")[-1]
        input_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_{self.modality}.nii.gz")
        target_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_tumor_segmentation.nii.gz")
        input_image = self._load_nifti_image(input_path)
        target_image = self._load_nifti_image(target_path)

        # Center crop the input and target images
        input_image = self._center_crop(input_image, self.crop_size)
        target_image = self._center_crop(target_image, self.crop_size)

        # Normalize input image based on its max value
        max_value = np.max(input_image)
        normalized_input = input_image / max_value

        # Convert input and target images to float32 ()
        normalized_input = normalized_input.astype(np.float32)
        target_image = target_image.astype(np.float32)

        # Map intensity values to class indices
        intensity_to_class = {
            0: 0, # Background
            2: 1, # Outer tumor region
            4: 2, # Enhancing tumor
            1: 3 # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        target_image = map_func(target_image).astype(np.int32)

        # Add 1 channel at index 0 to input and target image to match [batch_size, channels, depth, height, width] expectations by PyTorch
        normalized_input = np.expand_dims(normalized_input, axis=0)
        target_image = np.expand_dims(target_image, axis=0)

        # Convert to PyTorch tensors
        normalized_input = torch.from_numpy(normalized_input).float() # Will convert to float 32 which is enough precision
        target_image = torch.from_numpy(target_image).long()

        # Apply transform if provided
        if self.transform:
            normalized_input = self.transform(normalized_input)

        return normalized_input, target_image

    def _load_nifti_image(self, path):
        image = nib.load(path).get_fdata()
        return image

    def _center_crop(self, image, crop_size):
        depth, height, width = image.shape
        crop_depth, crop_height, crop_width = crop_size

        start_depth = (depth - crop_depth) // 2
        start_height = (height - crop_height) // 2
        start_width = (width - crop_width) // 2

        cropped_image = image[start_depth:start_depth+crop_depth, start_height:start_height+crop_height, start_width:start_width+crop_width]
        return cropped_image

def visualize_example(data_dir):

    train_folders, val_folders, test_folders = MRIDataset.split_data(config.data_dir)

    dataset = MRIDataset(data_dir, train_folders)
    input_image, target_image = dataset[0]

    # Convert PyTorch tensors back to numpy arrays for visualization
    input_image = input_image.numpy().squeeze()
    target_image = target_image.numpy().squeeze()

    # Load the original input image without cropping
    patient_folder = dataset.patient_folders[0]
    patient_number = patient_folder.split("_")[0].split("-")[-1]
    original_input_path = os.path.join(data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_FLAIR_bias.nii.gz")
    print(original_input_path)
    original_input_image = dataset._load_nifti_image(original_input_path)

    # Visualize the original and cropped input image
    plt.subplot(1, 2, 1)
    plt.imshow(original_input_image[:, original_input_image.shape[2] // 2, :], cmap='gray')
    plt.title("Original Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(input_image[:, input_image.shape[2] // 2, :], cmap='gray')
    plt.title("Cropped Input Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/evaluate_data/"
    visualize_example(data_dir)