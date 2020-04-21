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
import torchvision
import torchvision.transforms as transforms 
#seed = 1
#torch.manual_seed(seed)

## Inter-project imports
from utils import train_network
from utils import parameters
from utils import datasets
from utils import evaluate_network
from utils.transforms import RotationTransform
from utils.transforms import GammaJitter
from utils.transforms import RandomScale
from utils.transforms import Resize
from utils.transforms import AdaptiveCenterCrop
from utils.transforms import BrightnessJitter
from utils.transforms import MedianFilter
from utils.transforms import RGBNIRTransform
from utils.functions import listdir
from networks.resnet18_bn import resnet18 as resnet_rgb
from networks.resnet18_bn_nir import resnet18 as resnet_nir
from networks.resnet18_bn_rgbnir import resnet18 as resnet_rgbnir

if __name__ == '__main__':
    if sys.platform == 'win32':
        config_path = '/Users/frank.580/Desktop/code/cse-fabe/config/logan_pc.yaml'
    elif sys.platform == 'darwin':
        config_path = '/Users/loganfrank/Desktop/code/mlcv-final-project/code/config/logan_mac.yaml'
    elif sys.platform == 'linux':
        pass
    dataset = input('Please enter the crop you want to use: ')

    pool = 'avg'

    # Open the yaml config file
    try:
        with open(os.path.abspath(config_path)) as config_file: 
            config = yaml.safe_load(config_file)

            # Location of root directory of all images
            image_directory = config['Paths']['image_directory']

            # Location of network parameters (network state dicts, etc.)
            network_directory = config['Paths']['network_directory']

            # Location of parsed data (dataframes, etc.)
            data_directory = config['Paths']['data_directory']

            # Location of saved results from evaluation (confusion matrix, etc.)
            results_directory = config['Paths']['results_directory']

    except:
        raise Exception('Error loading data from config file.')

    # Define the compute device (either GPU or CPU)
    if torch.cuda.is_available():
        compute_device = torch.device('cuda:0')
        # torch.cuda.manual_seed_all(seed)
    else:
        compute_device = torch.device('cpu')
    print(f'Using device: {compute_device}')

    # Load in parameters from training
    # with open(f'{network_directory}{experiment}_parameters.pkl', 'rb') as f:
    #     parameters = pickle.load(f)

    # Do we want to use a balanced test and val dataset?
    balanced = True

    # Load pandas dataframe
    dataframe = pd.read_pickle(os.path.abspath(f'{data_directory}rgb.pkl'))

    # Create the data transforms for evaluating
    rgb_test_transform = transforms.Compose([Resize(size=256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])                       
    nir_test_transform = transforms.Compose([Resize(size=256), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.25])])
    rgbnir_test_transform = RGBNIRTransform(resize=256, normalize=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], train=False)

    if dataset == 'rgb':
        transform = rgb_test_transform
    elif dataset == 'nir':
        transform = nir_test_transform
    elif dataset == 'rgbnir':
        transform = rgbnir_test_transform

    if balanced:
        # Load testing dataset and dataloader
        test_dataset = datasets.balance_dataset(image_root_directory=image_directory, dataframe=dataframe,  
                                                transform=transform, phase='test', balance=True, cut=1, mode=dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load validation dataset and dataloader
        val_dataset = datasets.balance_dataset(image_root_directory=image_directory, dataframe=dataframe, 
                                                transform=transform, phase='val', balance=True, cut=1, mode=dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        # Create the test dataset
        test_dataset = datasets.imbalance_dataset(image_root_directory=image_directory, dataframe=dataframe, transform=transform, phase='test', mode=dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create the validation dataset
        val_dataset = datasets.imbalance_dataset(image_root_directory=image_directory, dataframe=dataframe, transform=transform, phase='val', mode=dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the network, load network state dictionary, and send the network to the compute device
    num_classes = len(test_dataset.classes_unique)
    if dataset == 'rgb':
        network = resnet_rgb(num_classes)
    elif dataset == 'nir':
        network = resnet_nir(num_classes)
    else:
        network = resnet_rgbnir(num_classes)
    
    network.load_state_dict(torch.load(os.path.abspath(f'{network_directory}{dataset}.pth'), map_location='cpu'))
    network = network.to(compute_device)
    network.eval()

    # Call a helper function to evaluate the neural network for validation and test sets
    print('Evaluating validation set:')
    evaluate_network.validate_baseline_network(network=network, dataloader=val_dataloader, compute_device=compute_device, 
                                                            experiment=dataset, results_directory=results_directory, 
                                                            classification_loss_func=nn.CrossEntropyLoss(), get_confusion_matrix=True)

    print('Evaluating test set:')
    evaluate_network.test_baseline_network(network=network, dataloader=test_dataloader, compute_device=compute_device, 
                                        experiment=dataset, results_directory=results_directory, save_results=True)