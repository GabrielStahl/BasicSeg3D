import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class MRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patient_folders = self._get_patient_folders()

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, index):
        patient_folder = self.patient_folders[index]
        patient_number = patient_folder.split("_")[0].split("-")[-1]

        input_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_T2_bias.nii.gz")
        target_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_tumor_segmentation.nii.gz")

        input_image = self._load_nifti_image(input_path)
        target_image = self._load_nifti_image(target_path)

        # Normalize input image based on its max value
        max_value = np.max(input_image)
        normalized_input = input_image / max_value

        # Convert input and target images to float32
        normalized_input = normalized_input.astype(np.float32)
        target_image = target_image.astype(np.float32)

        # Map intensity values to class indices: CE loss will expect target values to be between [0,3] for 4 classes
        intensity_to_class = {
            0: 0,  # Background
            2: 1,  # Outer tumor region
            4: 2,  # Enhancing tumor
            1: 3   # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        target_image = map_func(target_image).astype(np.int32)

        # Add 1 channel at index 1 to input and target image to match [batch_size, channels, depth, height, width] expectations by PyTorch
        normalized_input = np.expand_dims(normalized_input, axis=0)
        target_image = np.expand_dims(target_image, axis=0)

        # Convert to PyTorch tensors
        normalized_input = torch.from_numpy(normalized_input).float()
        target_image = torch.from_numpy(target_image).long() 

        return normalized_input, target_image

    def _get_patient_folders(self):
        return [folder for folder in os.listdir(self.data_dir) if folder.startswith("UCSF-PDGM-")]

    def _load_nifti_image(self, path):
        image = nib.load(path).get_fdata()
        return image

def get_dataloader(data_dir, batch_size=1, shuffle=True, num_workers=0):
    dataset = MRIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/" 