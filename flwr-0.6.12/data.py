import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib
from torch.utils.data import DataLoader, TensorDataset
from medmnist.dataset import PathMNIST, DermaMNIST
from torchvision.datasets import CIFAR10, SVHN, CIFAR100
import torchvision.transforms as transforms
import numpy as np
from utils import *
import open_clip
import faiss
from collections import Counter
import time
from scipy.stats import mode

def get_transforms(dataset_name, augmentation=True):
    """Returns the appropriate transformations based on the dataset name."""
    
    MEAN = {'cifar10': [0.485, 0.456, 0.406], 'cifar100': [0.507, 0.487, 0.441], 'tiny-imagenet': [0.480, 0.448, 0.398]}
    STD = {'cifar10': [0.229, 0.224, 0.225], 'cifar100': [0.267, 0.256, 0.276], 'tiny-imagenet': [0.277, 0.269, 0.282]}

    if augmentation:
        if dataset_name.lower() == "tiny-imagenet":
            data_transform = [
                            transforms.RandomCrop(64, padding=4),
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
        elif dataset_name.lower() in ["cifar10", "cifar100", "svhn"]:
            data_transform = [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
        else:
            data_transform = [
                            transforms.RandomCrop(28, padding=4),
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
    else:
        data_transform = [transforms.ToTensor(), 
                          transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]

    return transforms.Compose(data_transform)

def extract_labels(dataset):
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset does not have a known attribute for labels")

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, np.ndarray):
        pass
    else:
        labels = np.array(labels)
    labels = labels.flatten()
    return labels

def get_embeddings(dataset, model, device, fname, batch_size=64, save_path=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Check if the model has 'encode_image' method
            if hasattr(model, 'encode_image'):
                embeddings = model.encode_image(images).float()
            else:
                embeddings = model(images).float()
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    all_labels = all_labels.cpu().numpy()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(all_embeddings, save_path+fname)
    return all_embeddings

def get_data(dataset_name="cifar10", id=0, num_clients=10, return_eval_ds=False, batch_size=128, embed_input=False, encoder="SigLIP",
             split=None, alpha=None, num_workers=4, seed=0, data_dir="./data"):

    np.random.seed(seed)

    # load the OpenCLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if encoder == "SigLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-512', 'webli')
    elif encoder == "RN50":
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', 'openai')
    elif encoder == "CLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', 'openai')
    elif encoder == "EvaCLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('EVA02-L-14', 'merged2b_s4b_b131k')
    elif encoder == "DINOv2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Define the preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    model.to(device)
    model.eval()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # Path and file names for embeddings and labels
    save_path = "./embeddings/"

    # Define filenames for embeddings and labels
    train_embeddings_fname = f"{dataset_name}_embeddings_train.pt"
    test_embeddings_fname = f"{dataset_name}_embeddings_test.pt"
    
    # Choose dataset based on the provided name
    if dataset_name.lower() == "cifar10":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = CIFAR10(root=data_dir, train=True, download=True, transform=preprocess)
            test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = CIFAR10(root=data_dir, train=False, download=True, transform=preprocess)
        num_classes = 10

    elif dataset_name.lower() == "cifar100":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = CIFAR100(root=data_dir, train=True, download=True, transform=preprocess)
            test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = CIFAR100(root=data_dir, train=False, download=True, transform=preprocess)
        num_classes = 100
    
    elif dataset_name.lower() == "tiny-imagenet":
        with contextlib.redirect_stdout(None):
            train_dataset = TinyImageNet(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = TinyImageNet(root=data_dir, split="train", download=True, transform=preprocess)
            test_dataset = TinyImageNet(root=data_dir, split="val", download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = TinyImageNet(root=data_dir, split="val", download=True, transform=preprocess)
        num_classes = 200
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Utilize foundation model to encode all the training data
    # Handle embedding of dataset if embed_input is True
    if embed_input:
        # Check if test embeddings already exist
        if os.path.exists(os.path.join(save_path, train_embeddings_fname)):
            train_embeddings = torch.load(os.path.join(save_path, train_embeddings_fname))
        else:
            train_embeddings = get_embeddings(
                train_dataset_for_embeddings, model, device, 
                fname=train_embeddings_fname,
                batch_size=batch_size, 
                save_path=save_path
            )
        
        if os.path.exists(os.path.join(save_path, test_embeddings_fname)):
            test_embeddings = torch.load(os.path.join(save_path, test_embeddings_fname))
        else:
            # Create a subset of the test_dataset to encode all test samples
            test_subset = torch.utils.data.Subset(test_dataset_for_embeddings, list(range(len(test_dataset_for_embeddings))))
            test_embeddings = get_embeddings(
                test_subset, model, device, 
                fname=test_embeddings_fname, 
                batch_size=batch_size, 
                save_path=save_path
            )

    # Return evaluation dataset if required
    if return_eval_ds:
        if embed_input:
            test_labels = extract_labels(test_dataset)
            test_dataset_embeddings = CustomTensorDataset(test_embeddings, test_labels)
            eval_loader = DataLoader(test_dataset_embeddings, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
            num_samples = len(test_dataset_embeddings)
        else:
            eval_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
            num_samples = len(test_dataset)
        return eval_loader, num_classes, num_samples
    else:
        if split == "dir_balance":
            # Call dir_balance function
            clients_data, sample = dir_balance(
                dataset=train_dataset,
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_users=num_clients,
                alpha=alpha,
                data_dir=data_dir,
                sample=None
            )
            # Get the train indices for the specific client
            train_indices = clients_data[int(id)]
        else:
            split_fn = get_split_fn(split)
            # Split data into client-specific subsets
            train_indices = split_fn(idxs=extract_labels(train_dataset), num_shards=num_clients,
                                    num_samples=len(train_dataset), num_classes=num_classes, seed=seed)[int(id)]
        data_ratio = len(train_indices) / len(train_dataset)

        if embed_input:
            train_embeddings = torch.load(os.path.join(save_path, train_embeddings_fname)).float()
    
            all_labels = torch.tensor(extract_labels(train_dataset), dtype=torch.long)
            subset_embeddings = train_embeddings[train_indices]
            subset_labels = all_labels[train_indices]
            
            # Create EmbeddingDataset and Dataloader for the client's data
            train_dataset_embeddings = CustomTensorDataset(subset_embeddings, subset_labels)
            train_loader = DataLoader(train_dataset_embeddings, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            num_samples = len(train_indices)
        else:
            # Create a subset of the train dataset for the client
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            num_samples = len(train_indices)

        return train_loader, num_classes, num_samples, data_ratio
