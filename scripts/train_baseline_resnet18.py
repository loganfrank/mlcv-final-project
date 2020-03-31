## Basic Python libraries
import sys
import os
import argparse
import yaml
import pickle
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
np.random.seed(1)
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

## Inter-project imports
from utils import train_network
from utils import evaluate_network
from utils import datasets
from utils import parameters
from utils.transforms import RotationTransform
from utils.transforms import GammaJitter
from utils.transforms import RandomScale
from utils.transforms import Resize
from utils.transforms import AdaptiveCenterCrop
from utils.transforms import BrightnessJitter
from utils.transforms import MedianFilter
from utils.functions import listdir
from networks.resnet18_bn_nir import resnet18 as resnet

if __name__ == '__main__':
    if sys.platform == 'win32':
        config_path = '/Users/frank.580/Desktop/code/cse-fabe/config/logan_pc.yaml'
    elif sys.platform == 'darwin':
        config_path = '/Users/loganfrank/Desktop/research/agriculture/code/cse-fabe/config/logan_mac.yaml'
    elif sys.platform == 'linux':
        config_path = '/home/loganfrank/Desktop/code/mlcv-agriculture/code/config/logan_pc.yaml'
    dataset = input('Please enter the classifier name: ')
    dataset_image_directory = f'image_directory'
    dataset_network_directory = f'network_directory'
    dataset_data_directory = f'data_directory'
    dataset_results_directory = f'results_directory'

    batch_size = 30
    learning_rate = 0.01
    num_epochs = 20
    experiment = f'resnet18_{dataset}_batchsize{batch_size}_lr{learning_rate}_numepochs{num_epochs}'

    # Open the yaml config file
    try:
        with open(os.path.abspath(config_path)) as config_file: 
            config = yaml.safe_load(config_file)

            # Location of root directory of all images
            image_directory = config['Paths'][dataset_image_directory]

            # Location of network parameters (network state dicts, etc.)
            network_directory = config['Paths'][dataset_network_directory]

            # Location of parsed data (dataframes, etc.)
            data_directory = config['Paths'][dataset_data_directory]

            # Location of saved results from evaluation (confusion matrix, etc.)
            results_directory = config['Paths'][dataset_results_directory]

    except:
        raise Exception('Error loading data from config file.')

    # Do we want to balance the sets?
    balanced = True

    # Flag for if we want to load the network and continue training or fine tune
    load_network_flag = False
    load_weights = False

    if not load_network_flag:
        # This helps to ensure we don't overwrite existing results
        if(os.path.exists(f'{network_directory}{experiment}.pth')):
            files = listdir(network_directory, f'{experiment}_[0-9].pth')

            # If we now need to switch to counting in double digits
            if len(files) == 9:
                files = files + listdir(network_directory, f'{experiment}_[0-9][0-9].pth')
            
            # Add the num to the end of the file
            num = len(files) + 1
            experiment = f'{experiment}_{num}'

    # Define the compute device (either GPU or CPU)
    compute_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set up a parameters object for saving hyperparameters, etc.
    parameters = parameters.Parameters()
    if load_network_flag:
        with open(os.path.abspath(f'{network_directory}{experiment}_parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)
    else:
        parameters.experiment = experiment
        parameters.batch_size = batch_size
        parameters.learning_rate = learning_rate
        parameters.num_epochs = num_epochs

    # Load pandas dataframe
    dataframe = pd.read_pickle(os.path.abspath(f'{data_directory}rgb.pkl'))

    # Create the data transforms for each respective set
    rgb_train_transform = transforms.Compose([Resize(size=256), transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.RandomVerticalFlip(p=0.5), RotationTransform(angles=[0, 90, 180, 270]), 
                                        GammaJitter(low=0.9, high=1.1),
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])                         
    rgb_test_transform = transforms.Compose([Resize(size=256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    nir_train_transform = transforms.Compose([Resize(size=256), transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.RandomVerticalFlip(p=0.5), RotationTransform(angles=[0, 90, 180, 270]), 
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.25])])                         
    nir_test_transform = transforms.Compose([Resize(size=256), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.25])])
    rgbnir_train_transform = 5
    rgbnir_test_transform = 6

    if dataset == 'rgb':
        train_transform = rgb_train_transform
        test_transform = rgb_test_transform
    elif dataset == 'nir':
        train_transform = nir_train_transform
        test_transform = nir_test_transform
    elif dataset == 'rgbnir':
        train_transform = rgbnir_train_transform
        test_transform = rgbnir_test_transform


    if balanced:
        # Load training dataset and dataloader
        train_dataset = datasets.balance_dataset(image_root_directory=image_directory, dataframe=dataframe,  
                                                transform=train_transform, phase='train', balance=True, cut=None, mode=dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Load testing dataset and dataloader
        test_dataset = datasets.balance_dataset(image_root_directory=image_directory, dataframe=dataframe,  
                                                transform=test_transform, phase='test', balance=True, cut=1, mode=dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load validation dataset and dataloader
        val_dataset = datasets.balance_dataset(image_root_directory=image_directory, dataframe=dataframe, 
                                                transform=test_transform, phase='val', balance=True, cut=1, mode=dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        # Create the train dataset
        train_dataset = datasets.imbalance_dataset(image_root_directory=image_directory, dataframe=dataframe, transform=train_transform, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create the test dataset
        test_dataset = datasets.imbalance_dataset(image_root_directory=image_directory, dataframe=dataframe, transform=test_transform, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create the validation dataset
        val_dataset = datasets.imbalance_dataset(image_root_directory=image_directory, dataframe=dataframe, transform=test_transform, phase='val')
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = len(train_dataset.classes_unique)
    network = resnet(num_classes)
    if load_weights:
        network.load_state_dict(torch.load(os.path.abspath(f'{network_directory}resnet18_weights.pth'), map_location='cpu'))
    elif load_network_flag:
        network.load_state_dict(torch.load(os.path.abspath(f'{network_directory}{experiment}.pth'), map_location='cpu'))
    else:
        torch.save(network.state_dict(), os.path.abspath(f'{network_directory}{experiment}_initial_weights.pth'))
    
    # Ensure all parameters allow for gradient descent
    for parameter in network.parameters():
        parameter.requires_grad = True
    
    # If we can use multiple GPUs, do so
    """
    if torch.cuda.device_count() > 1 and batch_size > 1:
        print('Converting to DataParallel network')
        network = nn.DataParallel(network)
        parameters.parallel = True
    """

    # Send to GPU
    network = network.to(compute_device)

    # Create the optimizer and (potentially) load the optimizer state dictionary
    optimizer = optim.SGD(network.parameters(), lr=parameters.learning_rate, momentum=parameters.momentum, weight_decay=parameters.weight_decay)
    if load_network_flag:
        optimizer.load_state_dict(torch.load(os.path.abspath(f'{network_directory}{parameters.experiment}_optimizer.pth')))

    # Create a learning rate scheduler -- this will reduce the learning rate by a factor when learning becomes stagnant
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Send network and other parameters to a helper function for training the neural network
    train_network.train_balanced_network(network=network, optimizer=optimizer, scheduler=scheduler, 
                                        parameters=parameters, train_dataloader=train_dataloader, 
                                        val_dataloader=val_dataloader, compute_device=compute_device, 
                                        network_directory=network_directory, results_directory=results_directory)