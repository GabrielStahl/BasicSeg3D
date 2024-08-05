import torch
from torch.utils.data import DataLoader
from data_loader import MRIDataset
from model import UNet
import config
from utils import calculate_metrics_EVAL
import os
import tqdm

def evaluate(model, val_dataloader, device):
    # Evaluate on the validation set
    model.eval()
    val_metrics = {
        'avg_precision': 0.0,
        'avg_recall': 0.0,
        'avg_f1': 0.0,
        'avg_dice': 0.0,
        'precision_background': 0.0,
        'precision_outer_tumour': 0.0,
        'precision_enhancing_tumour': 0.0,
        'precision_tumour_core': 0.0,
        'recall_background': 0.0,
        'recall_outer_tumour': 0.0,
        'recall_enhancing_tumour': 0.0,
        'recall_tumour_core': 0.0,
        'f1_background': 0.0,
        'f1_outer_tumour': 0.0,
        'f1_enhancing_tumour': 0.0,
        'f1_tumour_core': 0.0,
        'dice_background': 0.0,
        'dice_outer_tumour': 0.0,
        'dice_enhancing_tumour': 0.0,
        'dice_tumour_core': 0.0
    }

    patient_dice_scores = {}

    with torch.no_grad():
        for inputs, targets, patient_id in tqdm.tqdm(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs) # torch.Size([1, 4, 150, 180, 116]) with logits for 4 classes
            targets = torch.squeeze(targets, 0) # squeeze the batch dimension, retain dim with class indices [0, 1, 2, 3], see intensity_to_class in data_loader.py

            predicted_labels = torch.argmax(outputs.detach(), dim=1) # now torch.Size([1, 150, 180, 116]) with class indices of maximum logits
            metrics = calculate_metrics_EVAL(predicted_labels, targets) # returns all metrics values across all classes

            for key, value in metrics.items():
                val_metrics[key] += value

            # Calculate average Dice score for this patient
            patient_avg_dice = (metrics['dice_background'] + metrics['dice_outer_tumour'] + 
                                metrics['dice_enhancing_tumour'] + metrics['dice_tumour_core']) / 4
            patient_dice_scores[patient_id[0]] = patient_avg_dice

    num_samples = len(val_dataloader)
    for key in val_metrics.keys():
        val_metrics[key] /= num_samples

    return val_metrics, patient_dice_scores


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    test_folders = [f for f in os.listdir(config.test_dir) if f.startswith("UCSF-PDGM-")]

    print(f"Evaluating performance on patients in: {config.test_dir}")
    print(f"Number of validation patients: {len(test_folders)}")

    modality = "T1c_bias"  # Choose from: "T1c_bias", "DTI_eddy_FA", "FLAIR_bias"
    print(f"Evaluating performance on modality: {modality}")

    # Load the dataset
    dataset = MRIDataset(config.test_dir, modality=modality)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)

    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        model_save_path = os.path.join(config.ensemble_path, f"{modality}_model_0.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded trained model weights from: {config.model_save_path}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path}")
        return

    # Evaluate the model
    val_metrics, patient_dice_scores = evaluate(model, dataloader, device)

    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # Find best and worst performing patients
    best_patient = max(patient_dice_scores, key=patient_dice_scores.get)
    worst_patient = min(patient_dice_scores, key=patient_dice_scores.get)

    # Print best and worst patient Dice scores
    print("\nBest and Worst Patient Dice Scores:")
    print(f"Best - Patient {best_patient}: {patient_dice_scores[best_patient]:.4f}")
    print(f"Worst - Patient {worst_patient}: {patient_dice_scores[worst_patient]:.4f}")


if __name__ == "__main__":
    main()