## Basic Python libraries
import sys
import os
import argparse
import yaml
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 

## Inter-project Imports
from utils.functions import listdir

def create_dataframe(path, output_path, dataset):

    # Initialize DataFrame with its index and column names
    index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['phase', 'field', 'instance'])
    dataframe = pd.DataFrame(index=index, columns=['anomaly'])

    # Loop through each set of data (i.e. train, test, val)
    phases = os.listdir(path)
    for phase in phases:
        phase_path = f'{path}{phase}/'

        # Skip over files (we only want directories at this point)
        if not os.path.isdir(phase_path):
            print(f'{phase_path} is not a directory!')
            continue

        

        # Loop through each disease in the crop (i.e. frogeye, rotten core, etc.)
        diseases = os.listdir(phase_path)
        for disease in diseases:
            disease_path = f'{phase_path}{disease}/'

            # Skip over files (we only want directories at this point)
            if not os.path.isdir(disease_path):
                print(f'{disease_path} is not a directory!')
                continue

            # Loop through each instance in the disease, add the instance as a row in the DataFrame
            instances = listdir(disease_path, '*.jpg')
            for instance in instances:
                if instance[-4:] != '.jpg' or instance[-8:] == '_nir.jpg':
                    print(f'We do not want this garbage: {instance}')
                    continue
                dataframe.loc[(phase, 'rgb', instance)] = disease
                    

    # Save the DataFrame for later use in constructing our DataSet and DataLoader objects
    dataframe.to_pickle(os.path.abspath(f'{output_path}{dataset}.pkl'))

if __name__ == '__main__':
    if sys.platform == 'win32':
        config_path = '/Users/frank.580/Desktop/code/cse-fabe/config/logan_pc.yaml'
    elif sys.platform == 'darwin':
        config_path = '/Users/loganfrank/Desktop/research/agriculture/code/cse-fabe/config/logan_mac.yaml'
    elif sys.platform == 'linux':
        config_path = '/home/loganfrank/Desktop/code/mlcv-agriculture/code/config/logan_pc.yaml'
    dataset = 'rgb'
    dataset_image_directory = f'image_directory'
    dataset_network_directory = f'network_directory'
    dataset_data_directory = f'data_directory'
    dataset_results_directory = f'results_directory'

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

    # Call a helper function to create, fill, and save a DataFrame with information about our data
    create_dataframe(image_directory, data_directory, dataset)