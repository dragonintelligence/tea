import os
import torch
import random
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils import data
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc, load_imagenet3dcc
from robustbench.data import load_cifar10, load_cifar100
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CIFARCDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        
        # Convert image from numpy to PIL if using torchvision transforms
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
def set_transform(dataset):
    if dataset.lower() == 'cifar10' or dataset.lower() == 'cifar100':
        # transform_train = T.Compose([ 
        #     T.Resize(32), 
        #     T.RandomCrop(32, padding=4), 
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor()
        #     ])
        # transform_test = T.Compose([T.Resize(32), T.ToTensor()])
        
        if dataset.lower() == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0.5070, 0.4865, 0.4409]
            std = [0.2673, 0.2564, 0.2761]
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
    elif  dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            T.Resize(28), 
            T.ToTensor(),
            #T.Normalize((0.1307,), (0.3081,)) 
        ])
        transform_test = T.Compose([T.Resize(28), T.RandomRotation(degrees=(90,91)), T.ToTensor()])
    elif  'tin200' in dataset.lower():
        transform_train = transforms.Compose([
            T.Resize(32), 
            T.RandomCrop(32, padding=4), 
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([T.Resize(32), T.ToTensor()])
    elif  'pacs' in dataset.lower():
        transform_train = transforms.Compose([
            T.Resize(32), 
            T.RandomCrop(32, padding=4), 
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([T.Resize(32), T.ToTensor()])
    else:
        raise
    return transform_train, transform_test

def load_tin200(n_examples, severity=None, data_dir=None, shuffle=False, corruptions=None, transform=None):
    if corruptions is not None:
        for corruption in corruptions:
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'Tiny-ImageNet-C', corruption, str(severity)), transform=transform)
    else:
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'val'), transform=transform)
    
    batch_size = 100
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor

def load_pacs(data_dir=None, shuffle=False, corruptions=None, transform=None):
    
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'pacs', corruptions), transform=transform) 
    
    batch_size = 100
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)

    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    return x_test_tensor, y_test_tensor

def load_data(data, n_examples=None, severity=None, data_dir=None, shuffle=False, corruptions=None):
        data_dir = os.path.join(".", "data")
        shuffle = True
        if data == 'cifar10':
            _, transform = set_transform(data)
            # x_test, y_test = load_cifar10(n_examples, data_dir=data_dir)
            valset = CIFAR10(data_dir, train=False, transform=transform, download=True)
            num_workers = 4
            x_test_ = DataLoader(valset, batch_size=64, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
            x_test = torch.cat([x[0] for x in x_test_], dim=0)
            y_test = torch.cat([x[1] for x in x_test_], dim=0)
            if n_examples is not None:
                x_test = x_test[:n_examples]
                y_test = y_test[:n_examples]
        elif data == 'cifar100':
            x_test, y_test = load_cifar100(n_examples, data_dir=data_dir)
            _, transform = set_transform(data)
            valset = CIFARCDataset(x_test, y_test, transform=transform)
            num_workers = 4
            x_test_ = DataLoader(valset, batch_size=64, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
            x_test = torch.cat([x[0] for x in x_test_], dim=0)
            y_test = torch.cat([x[1] for x in x_test_], dim=0)
            if n_examples is not None:
                x_test = x_test[:n_examples]
                y_test = y_test[:n_examples]
        elif data == 'tin200':
            _, transform = set_transform(data)
            x_test, y_test = load_tin200(n_examples=n_examples, data_dir=data_dir, transform=transform)
        elif data == 'cifar10c':
            x_test, y_test = load_cifar10c(n_examples, severity, data_dir, shuffle, corruptions)
            _, transform = set_transform('cifar10')
            print("SHUFFLE ", shuffle)
            valset = CIFARCDataset(x_test, y_test, transform=transform)
            num_workers = 4
            x_test_ = DataLoader(valset, batch_size=64, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
            x_test = torch.cat([x[0] for x in x_test_], dim=0)
            y_test = torch.cat([x[1] for x in x_test_], dim=0)
            if n_examples is not None:
                x_test = x_test[:n_examples]
                y_test = y_test[:n_examples]
        elif data == 'cifar100c':
            print("SHUFFLE ", shuffle)
            x_test, y_test = load_cifar100c(n_examples, severity, data_dir, shuffle, corruptions)
            _, transform = set_transform('cifar100')
            valset = CIFARCDataset(x_test, y_test, transform=transform)
            num_workers = 4
            x_test_ = DataLoader(valset, batch_size=64, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
            x_test = torch.cat([x[0] for x in x_test_], dim=0)
            y_test = torch.cat([x[1] for x in x_test_], dim=0)
            if n_examples is not None:
                x_test = x_test[:n_examples]
                y_test = y_test[:n_examples]
        elif data == 'tin200c':
            _, transform = set_transform(data)
            x_test, y_test = load_tin200(n_examples=n_examples, severity=severity, data_dir=data_dir, shuffle=shuffle, corruptions=corruptions, transform=transform)
        elif data == 'pacs':
            _, transform = set_transform(data)
            x_test, y_test = load_pacs(data_dir=data_dir, shuffle=shuffle, corruptions=corruptions, transform=transform)
        
        print(x_test.shape, n_examples)
        return x_test, y_test

def load_dataloader(root, dataset, batch_size, if_shuffle, logger=None):
    train_transforms, test_transforms = set_transform(dataset)
    if dataset.lower() == 'cifar10':
        logger.info("using cifar10..")
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'cifar100':
        logger.info("using cifar100..")
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'mnist':
        logger.info("using mnist..")
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=test_transforms)
    elif dataset.lower() == 'tin200':
        logger.info("using tin200..")
        train_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'val'), transform=test_transforms)
    elif 'pacs' in dataset.lower():
        train_dataset = datasets.ImageFolder(os.path.join(root, 'pacs', dataset.split("-")[1]), transform=train_transforms) 
        test_dataset = datasets.ImageFolder(os.path.join(root, 'pacs', dataset.split("-")[1]), transform=test_transforms) 
    else:
        raise
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,  num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,  num_workers=4, shuffle=if_shuffle)
    return train_dataset, test_dataset, train_loader, test_loader

