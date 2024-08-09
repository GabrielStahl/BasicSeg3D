import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import config

class CorrectionDataset(Dataset):
    def __init__(self, data_subset, modality = "T1c_bias", UMap = "modality_ensemble", crop_size=config.crop_size, transform=None, test_mode=False):
        
        self.test_mode = test_mode # if true, don't try to load target images (error masks)
        
        self.data_subset = data_subset

        if data_subset == "train_set":
            self.data_dir = config.train_dir
        elif data_subset == "val_set":
            self.data_dir = config.val_dir
        elif data_subset == "test_set":
            self.data_dir = config.test_dir
        
        self.patient_folders = [folder for folder in os.listdir(self.data_dir) if folder.startswith("UCSF-PDGM-") and "FU" not in folder]
        self.crop_size = crop_size
        self.transform = transform
        self.modality = modality
        self.UMap = UMap

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, index):
        patient_folder = self.patient_folders[index]
        patient_number = patient_folder.split("_")[0].split("-")[-1]

        # Load original MRI, predicted segmentation, uncertainty map, and target segmentation
        mri_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_{self.modality}.nii.gz")
        pred_seg_path = os.path.join(config.root_dir, f"predictions_{self.data_subset}", self.UMap, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        uncertainty_path = os.path.join(config.root_dir, f"predictions_{self.data_subset}", self.UMap, f"{self.UMap}_UMap_UCSF-PDGM-{patient_number}.nii.gz")
        target_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_tumor_segmentation.nii.gz")

        mri_image = self._load_nifti_image(mri_path)
        pred_seg_image = self._load_nifti_image(pred_seg_path)
        uncertainty_image = self._load_nifti_image(uncertainty_path)
        target_image = self._load_nifti_image(target_path)

        # Center crop all images
        mri_image = self._center_crop(mri_image, self.crop_size)
        pred_seg_image = self._center_crop(pred_seg_image, self.crop_size)
        uncertainty_image = self._center_crop(uncertainty_image, self.crop_size)
        target_image = self._center_crop(target_image, self.crop_size)

        # Normalize MRI image
        mri_image = mri_image / np.max(mri_image)

        # Combine inputs
        input_image = np.stack([mri_image, pred_seg_image, uncertainty_image], axis=0) 

        # Convert input and target to float32 bc dl models like them
        input_image = input_image.astype(np.float32)
        target_image = target_image.astype(np.float32)

        # Map intensity values to class indices for predicted segmentation
        intensity_to_class = {
            0: 0,  # Background
            2: 1,  # Outer tumor region
            4: 2,  # Enhancing tumor
            1: 3   # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        pred_seg_image = map_func(pred_seg_image)
        target_image = map_func(target_image)

        # Convert to PyTorch tensors
        input_image = torch.from_numpy(input_image)
        target_image = torch.from_numpy(target_image)

        # Apply transform if provided
        if self.transform:
            input_image = self.transform(input_image)

        return input_image, target_image, patient_number

    def _load_nifti_image(self, path):
        return nib.load(path).get_fdata()

    def _center_crop(self, image, crop_size):
        depth, height, width = image.shape
        crop_depth, crop_height, crop_width = crop_size

        start_depth = (depth - crop_depth) // 2
        start_height = (height - crop_height) // 2
        start_width = (width - crop_width) // 2

        cropped_image = image[start_depth:start_depth+crop_depth,
                              start_height:start_height+crop_height,
                              start_width:start_width+crop_width]
        return cropped_image