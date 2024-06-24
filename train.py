import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_loader import MRIDataset
from model import UNet
import config
from utils import calculate_metrics, DiceLoss
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import os
import socket

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    running_dice = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            targets = torch.squeeze(targets, 1)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        predicted_labels = torch.argmax(outputs.detach(), dim=1)
        precision, recall, f1, dice = calculate_metrics(predicted_labels, targets)
        running_precision += precision
        running_recall += recall
        running_f1 += f1
        running_dice += dice

    epoch_loss = running_loss / len(train_dataloader)
    epoch_precision = running_precision / len(train_dataloader)
    epoch_recall = running_recall / len(train_dataloader)
    epoch_f1 = running_f1 / len(train_dataloader)
    epoch_dice = running_dice / len(train_dataloader)

    # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            targets = torch.squeeze(targets, 1)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            predicted_labels = torch.argmax(outputs.detach(), dim=1)
            precision, recall, f1, dice = calculate_metrics(predicted_labels, targets)
            val_precision += precision
            val_recall += recall
            val_f1 += f1
            val_dice += dice

    val_loss /= len(val_dataloader)
    val_precision /= len(val_dataloader)
    val_recall /= len(val_dataloader)
    val_f1 /= len(val_dataloader)
    val_dice /= len(val_dataloader)

    return epoch_loss, epoch_precision, epoch_recall, epoch_f1, epoch_dice, val_loss, val_precision, val_recall, val_f1, val_dice

def setup_DDP(rank, world_size):
    os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
    os.environ["MASTER_PORT"] = "33230"
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size) 

def main():
    # Device configuration
    environment = config.environment

    # Split the data into train, validation, and test sets
    train_folders, val_folders, test_folders = MRIDataset.split_data(config.data_dir)

    # Load the datasets
    train_dataset = MRIDataset(config.data_dir, train_folders)
    val_dataset = MRIDataset(config.data_dir, val_folders)
    test_dataset = MRIDataset(config.data_dir, test_folders)

    # Setup DDP and create distributed samplers if not in local environment
    if environment != 'local':

        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = local_rank # because we are using a single node

        setup_DDP(rank, world_size)

        device_id = local_rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, USING Device ID: {device_id}")

        # Create distributed samplers
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

        # Set samplers to None
        train_sampler = None
        val_sampler = None
        test_sampler = None

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, num_workers=0)

    # Create the model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model = model.to(device)

    # Wrap the model with DistributedDataParallel only if not in local environment
    if environment != 'local':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Loss function and optimizer
    criterion = DiceLoss()  
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create the GradScaler
    scaler = GradScaler()

    # Training loop
    for epoch in range(config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch) 

        #print(f"PyTorch CUDA version: {torch.version.cuda}")
        #os.system('nvidia-smi')

        epoch_loss, epoch_precision, epoch_recall, epoch_f1, epoch_dice, val_loss, val_precision, val_recall, val_f1, val_dice = train(model, train_dataloader, val_dataloader, optimizer, criterion, device, scaler, epoch)
        print(f"Epoch [{epoch+1}/{config.epochs}], "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Precision: {epoch_precision:.4f}, "
            f"Train Recall: {epoch_recall:.4f}, "
            f"Train F1: {epoch_f1:.4f}, "
            f"Train Dice: {epoch_dice:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Precision: {val_precision:.4f}, "
            f"Val Recall: {val_recall:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Val Dice: {val_dice:.4f}")
    
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
