"""Data loading and training/evaluation functions for FL"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from models import get_model


# Data transforms for CIFAR10
def apply_train_transforms(batch):
    """Apply data augmentation transforms for training"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch


def apply_eval_transforms(batch):
    """Apply transforms for evaluation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch


# Global FederatedDataset cache
fds = None


def get_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Load CIFAR10 data partition for federated learning.

    Args:
        partition_id: Client partition ID
        num_partitions: Total number of partitions
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (trainloader, testloader)
    """
    global fds

    if fds is None:
        # Create IID partitioner
        partitioner = IidPartitioner(num_partitions=num_partitions)

        # Load CIFAR10 dataset
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner}
        )

    # Load specific partition
    partition = fds.load_partition(partition_id)

    # Split into train/test (80/20)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Apply transforms
    train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)

    # Create DataLoaders
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_partition, batch_size=batch_size, drop_last=True)

    return trainloader, testloader


def train(model, trainloader, epochs: int, learning_rate: float, device):
    """Train the model on local data.

    Args:
        model: PyTorch model
        trainloader: Training data loader
        epochs: Number of local training epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Tuple of (loss, accuracy)
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        for batch in trainloader:
            # Handle batch format
            if isinstance(batch, dict):
                data, target = batch["img"], batch["label"]
            else:
                data, target = batch

            # Convert to tensors if needed
            if not isinstance(data, torch.Tensor):
                data = torch.stack(data) if isinstance(data, list) else data
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target) if not isinstance(target, torch.Tensor) else target

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

    avg_loss = total_loss / (len(trainloader) * epochs)
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def test(model, testloader, device):
    """Evaluate the model on test data.

    Args:
        model: PyTorch model
        testloader: Test data loader
        device: Device to evaluate on

    Returns:
        Tuple of (loss, accuracy)
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            # Handle batch format
            if isinstance(batch, dict):
                data, target = batch["img"], batch["label"]
            else:
                data, target = batch

            # Convert to tensors if needed
            if not isinstance(data, torch.Tensor):
                data = torch.stack(data) if isinstance(data, list) else data
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target) if not isinstance(target, torch.Tensor) else target

            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy
