import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_loader import MRIDataset
from model import UNet
import config
from utils import calculate_dice_coefficient
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import os

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            targets = torch.squeeze(targets, 1)  # Squeeze away the "channel" dimension in targets to get [N, D, H, W] (N being batch size)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        predicted_labels = torch.argmax(outputs.detach(), dim=1) # Convert logits to class indices before dice calculation
        running_dice += calculate_dice_coefficient(predicted_labels, targets)

    epoch_loss = running_loss / len(train_dataloader)
    epoch_dice = running_dice / len(train_dataloader)

    # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  
            targets = torch.squeeze(targets, 1)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            predicted_labels = torch.argmax(outputs.detach(), dim=1) 
            val_dice += calculate_dice_coefficient(predicted_labels, targets)

    val_loss /= len(val_dataloader)
    val_dice /= len(val_dataloader)

    return epoch_loss, epoch_dice, val_loss, val_dice

def setup_DDP(rank, world_size):
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Split the data into train, validation, and test sets
    train_folders, val_folders, test_folders = MRIDataset.split_data(config.data_dir)

    # Load the datasets
    train_dataset = MRIDataset(config.data_dir, train_folders)
    val_dataset = MRIDataset(config.data_dir, val_folders)
    test_dataset = MRIDataset(config.data_dir, test_folders)

    # Create distributed samplers if not in local environment
    environment = config.environment

    if environment != 'local':
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup_DDP(rank, world_size)
        local_rank = rank # This is always true as I only use 1 node
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if environment != 'local':
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, num_workers=0)

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)

    # Move the model to the appropriate device
    if environment != 'local':
        device_id = rank % torch.cuda.device_count()
        print(f"Using device id: {device_id}")
        model = model.to(device_id)
    else:
        model.to(device)
        print(f"Using device: {device}")
    
    # Wrap the model with DistributedDataParallel only if not in local environment
    if environment != 'local':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # We use cross-entropy loss for multi-class prediction
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create the GradScaler
    scaler = GradScaler()

    # Training loop
    for epoch in range(config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch) 
        epoch_loss, epoch_dice, val_loss, val_dice = train(model, train_dataloader, val_dataloader, optimizer, criterion, device, scaler, epoch)
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0 and (environment == 'local' or dist.get_rank() == 0):
            save_path = f"{config.model_save_path}epoch_{epoch+1}.pth"
            if environment != 'local':
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Save the trained model
    if environment == 'local' or dist.get_rank() == 0:
        save_path = f"{config.model_save_path}final_epoch.pth"
        if environment != 'local':
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    # Clean up the distributed environment if not in local environment
    if environment != 'local':
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
