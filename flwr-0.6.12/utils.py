import numpy as np
import time
import GPUtil
import os
import random
import math
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    check_integrity,
    download_url,
    extract_archive,
    verify_str_arg,
)

def none_or_str(value):
    if value == 'None':
        return None
    return value

def grab_gpu(memory_limit=0.91, max_wait_time=600):
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)

        if cuda_device_ids:
            return str(cuda_device_ids[0])
        else:
            print("Waiting for available GPU...")
            time.sleep(10)

    raise RuntimeError("No GPU available within the maximum wait time.")

def create_iid_shards(idxs, num_shards, num_samples, num_classes, seed=1):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_imbalanced_shards(idxs, num_shards, num_samples, num_classes, skewness=0.8, seed=1):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples, p=np.random.dirichlet(np.repeat(skewness, num_shards))).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_noniid_shards(idxs, num_shards, num_samples, num_classes, skewness=0.1, seed=1,):
    np.random.seed(seed)
    partitions = {}
    min_size = 0
    min_require_size = 10
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_shards)]
        for k in range(num_classes):
            idx_k = np.where(idxs==k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(skewness, num_shards))
            proportions = np.array([p * (len(idx_j) < num_samples / num_shards) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_shards):
            np.random.shuffle(idx_batch[j])
            partitions[j] = idx_batch[j]
    return partitions

def dir_balance(dataset, dataset_name, num_classes, num_users, alpha, data_dir, sample=None):
    """ for the fairness of annotation cost, each client has same number of samples
    """
    C = num_classes
    K = num_users
    alpha = alpha
    
    # Generate the set of clients dataset.
    clients_data = {}
    for i in range(K):
        clients_data[i] = []

    # Divide the dataset into each class of dataset.
    total_num = len(dataset)
    total_data = {}
    data_num = np.array([0 for _ in range(C)])
    for i in range(C):
        total_data[str(i)] = []
    for idx, data in enumerate(dataset):
        if dataset_name in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            total_data[str(data[1][0])].append(idx)
            data_num[int(data[1][0])] += 1
        else:
            total_data[str(data[1])].append(idx)
            data_num[int(data[1])] += 1

    clients_data_num = {}
    for client in range(K):
        clients_data_num[client] = [0] * C
    
    # Distribute the data with the Dirichilet distribution.
    if sample is None:
        diri_dis = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(C))
        sample = torch.cat([diri_dis.sample().unsqueeze(0) for _ in range(K)], 0)

        # get balanced matrix
        rsum = sample.sum(1)
        csum = sample.sum(0)
        epsilon = min(1 , K / C, C / K) / 1000

        if alpha < 10:
            r, c = 1, K / C
            while (torch.any(rsum <= r - epsilon)) or (torch.any(csum <= c - epsilon)):
                sample /= sample.sum(0)
                sample /= sample.sum(1).unsqueeze(1)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        else:
            r, c = C / K, 1
            while (torch.any(abs(rsum - r) >= epsilon)) or (torch.any(abs(csum - c) >= epsilon)):
                sample = sample / sample.sum(1).unsqueeze(1)
                sample /= sample.sum(0)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        
    x = sample * torch.tensor(data_num)
    x = torch.ceil(x).long()
    x = torch.where(x <= 1, 0, x+1) if alpha < 10 else torch.where(x <= 1, 0, x)
    # print(x)
    
    print('Dataset total num', len(dataset))
    print('Total dataset class num', data_num)

    if alpha < 10:
        remain = np.inf
        nums = math.ceil(len(dataset) / K)
        i = 0
        while remain != 0:
            i += 1
            for client_idx in clients_data.keys():
                for cls in total_data.keys():
                    tmp_set = random.sample(total_data[cls], min(len(total_data[cls]), x[client_idx, int(cls)].item()))
                    
                    if len(clients_data[client_idx]) + len(tmp_set) > nums:
                        tmp_set = tmp_set[:nums-len(clients_data[client_idx])]

                    clients_data[client_idx] += tmp_set
                    clients_data_num[client_idx][int(cls)] += len(tmp_set)

                    total_data[cls] = list(set(total_data[cls])-set(tmp_set))   

            remain = sum([len(d) for _, d in total_data.items()])
            if i == 100:
                break
                
        # to make same number of samples for each client
        index = np.where(np.array([sum(clients_data_num[k]) for k in clients_data_num.keys()]) <= nums-1)[0]
        for client_idx in index:
            n = nums - len(clients_data[client_idx])
            add = 0
            for cls in total_data.keys():
                tmp_set = total_data[cls][:n-add]
                
                clients_data[client_idx] += tmp_set
                clients_data_num[client_idx][int(cls)] += len(tmp_set)
                total_data[cls] = list(set(total_data[cls])-set(tmp_set))  
                
                add += len(tmp_set)
    else:
        cumsum = x.T.cumsum(dim=1)
        for cls, data in total_data.items():
            cum = list(cumsum[int(cls)].numpy())
            tmp = np.split(np.array(data), cum)

            for client_idx in clients_data.keys():
                clients_data[client_idx] += list(tmp[client_idx])
                clients_data_num[client_idx][int(cls)] += len(list(tmp[client_idx]))

    print('clients_data_num', clients_data_num)
    print('clients_data_num', [sum(clients_data_num[k]) for k in clients_data_num.keys()])
    with open(os.path.join(data_dir, 'clients_data_num.pickle'), 'wb') as f:
        pickle.dump(clients_data_num, f)

    return clients_data, sample

def get_split_fn(name='iid', **split_fn_kwargs):
    if name == 'iid':
        return create_iid_shards
    elif name == 'noniid':
        return create_noniid_shards
    elif name == 'imbalanced':
        return create_imbalanced_shards
    elif name == 'dir_balance':
        return ...
    else:
        raise ValueError("Invalid name provided. Supported names are 'iid', 'noniid', 'imbalanced', and 'dir_balance'.")

import math

def get_learning_rate(initial_lr, current_round, total_rounds, decay_factor=0.5, num_decays=3):
    """
    Step-wise decay of learning rate.
    """
    # Determine the number of decays that should have occurred by the current round
    decay_step = total_rounds / (num_decays + 1)  # +1 to include the initial LR at round 0
    num_applied_decays = int(current_round / decay_step)
    
    # Calculate the new learning rate
    new_lr = initial_lr * (decay_factor ** num_applied_decays)
    
    return new_lr

class TinyImageNet(VisionDataset):
    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.loader = default_loader

        if not self._check_integrity():
            if download:
                self._download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')
        else:
            print('Files already downloaded and verified.')

        # Extract if necessary
        if not os.path.isdir(self.dataset_folder):
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.filename), self.root)

        # Prepare data
        classes, class_to_idx = find_classes(os.path.join(self.dataset_folder, 'wnids.txt'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = make_dataset(self.dataset_folder, self.split, class_to_idx)
        self.targets = [target for _, target in self.data]  # Add targets attribute

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename, md5=self.md5)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename), self.root)

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

def find_classes(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    classes.sort()
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(root, split, class_to_idx):
    images = []
    if split == 'train':
        train_dir = os.path.join(root, 'train')
        for cls_name in os.listdir(train_dir):
            cls_dir = os.path.join(train_dir, cls_name, 'images')
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    item = (img_path, class_to_idx[cls_name])
                    images.append(item)
    elif split == 'val':
        val_dir = os.path.join(root, 'val')
        img_dir = os.path.join(val_dir, 'images')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')

        # Map image filenames to class names
        cls_map = {}
        with open(val_annotations, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                img_name, cls_name = tokens[0], tokens[1]
                cls_map[img_name] = cls_name

        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            cls_name = cls_map[img_name]
            item = (img_path, class_to_idx[cls_name])
            images.append(item)
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    return images

class CustomTensorDataset(Dataset):
    def __init__(self, data, targets):
        """
        Args:
            data (Tensor): Input data.
            targets (Tensor or list): Corresponding labels.
        """
        self.data = data
        self.targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)