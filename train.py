import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MRIDataset
from model import UNet
import config
from utils import calculate_dice_coefficient

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        print(f"Batch {batch_idx + 1} with batch size {inputs.size(0)} and image size {inputs.size()}")

        if torch.cuda.is_available():
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device.index) / (1024 ** 3):.2f} GB")
            print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device.index) / (1024 ** 3):.2f} GB")

        optimizer.zero_grad()
        outputs = model(inputs)
        targets = torch.squeeze(targets, 1) # Squeeze away the "channel" dimension in targets to get [N, D, H, W] (N being batch size)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += calculate_dice_coefficient(outputs.detach(), targets)


    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    return epoch_loss, epoch_dice

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    dataset = MRIDataset(config.data_dir)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss() # We use cross-entropy loss for multi-class prediction
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        epoch_loss, epoch_dice = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    main()