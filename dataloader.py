import torch
from torch.functional import split
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np

def prep_dataloader(args):


    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])

    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if args['dataset'] == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/cifar10',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR10(root='data/cifar10',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif args['dataset'] == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='data/cifar100',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

        test_dataset = datasets.CIFAR100(root='data/cifar100',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args['batch_size'],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args['batch_size'],
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)
    
    return train_loader, test_loader


def toy_dataloader(args):
    def synthetic_data(w, b, m_examples, scale=1):
        X = torch.randn((m_examples, len(w)))
        X[:, 1] *= scale
        y = torch.mv(X, w)
        y += torch.normal(0, 0.5, y.shape)
        return X, y

    def initial(config, w=None, b=0.0, scale=1):
        m, n = config["m"], config['n']
        return synthetic_data(w, b, m, scale)

    config = {
        "m": 200,
        "n": 2,
        "batch_size": 200,
        "lr": 0.15,
        "n_epochs": 60,
        "optimizer": "SGD",
        "optim_hparas": {
            'lr': 0.015,         # for SGD and Adam
            'momentum': 0.9,
            'nesterov':True
        }
    }
    true_w = torch.tensor([1., 1.])
    X, y = initial(config, true_w, scale=2.5)
    train_dataset = torch.utils.data.TensorDataset(X, y)

    splits = []
    size = len(train_dataset) // args['K']
    for _ in range(args['K'] - 1):
        splits.append(size)
    splits.append(len(train_dataset) - size * (args['K'] - 1))

    train_datasets = torch.utils.data.random_split(train_dataset, splits)

    # print(f'Dataset: {len(train_dataset)}/{len(test_dataset)}')

    train_loaders = [torch.utils.data.DataLoader(dataset, 200, shuffle=True,
                                num_workers=0) for dataset in train_datasets]
    test_loader = torch.utils.data.DataLoader(train_dataset, 200, shuffle=False,
                            num_workers=0)

    return train_loaders, test_loader
