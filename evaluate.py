import torch
from torch.utils.data import DataLoader
from data_loader import MRIDataset
from model import UNet
import config
from utils import calculate_dice_coefficient, calculate_metrics
import os

def evaluate(model, dataloader, device):
    model.eval()
    dice_scores = []
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)

            dice = calculate_dice_coefficient(predictions, targets)
            precision, recall, f1 = calculate_metrics(predictions, targets)

            dice_scores.append(dice)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    return avg_dice, avg_precision, avg_recall, avg_f1


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = MRIDataset(config.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)

    # Load the trained model weights
    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path))
        print(f"Loaded trained model weights from: {config.model_save_path}")
    else:
        print(f"Trained model weights not found at: {config.model_save_path}")
        return

    # Evaluate the model
    avg_dice, avg_precision, avg_recall, avg_f1 = evaluate(model, dataloader, device)

    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

if __name__ == "__main__":
    main()