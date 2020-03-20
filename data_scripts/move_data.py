## Basic Python libraries
import sys
import os
import shutil
import argparse
import yaml
import math
import glob
sys.path.append(os.getcwd() + '/')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def split_data(old_dir, train_dir, test_dir, val_dir):
    train_split = 0.7
    test_split = 0.15
    for class_index, class_name in enumerate(os.listdir(old_dir)):
        class_path = f'{old_dir}{class_name}/'
        if not os.path.isdir(class_path):
            print(f'{class_path} is not a directory!!')
            continue
        
        num_instances = os.listdir(class_path)
        num_instances = [x for x in num_instances if x[-4:] == '.jpg']
        num_instances = len(num_instances) // 2
        train_instances = math.floor(train_split * num_instances)
        test_instances = math.floor(test_split * num_instances)

        counter = 0
        for instance_index, instance_name in enumerate(os.listdir(class_path)):
            if instance_name[-4:] != '.jpg' or instance_name[-7:] == 'nir.jpg':
                print(f'{instance_name} is not an RGB jpg!')
                continue
            rgb_src = f'{class_path}{instance_name}'
            nir_src = f'{class_path}{instance_name}'.replace('.jpg', '_nir.jpg')

            train_path = f'{train_dir}{class_name}/'
            test_path = f'{test_dir}{class_name}/'
            val_path = f'{val_dir}{class_name}/'

            if counter < train_instances:
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                shutil.copy(rgb_src, train_path)
                shutil.copy(nir_src, train_path)
            elif counter < train_instances + test_instances:
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                shutil.copy(rgb_src, test_path)
                shutil.copy(nir_src, test_path)
            else:
                if not os.path.exists(val_path):
                    os.makedirs(val_path)
                shutil.copy(rgb_src, val_path)
                shutil.copy(nir_src, val_path)

            counter += 1

def split_unsupervised(old_dir, train_path, test_path, val_path):
    train_split = 0.7
    test_split = 0.15
    num_instances = os.listdir(old_dir)
    num_instances = [x for x in num_instances if x[-4:] == '.jpg']
    num_instances = len(num_instances) // 2
    train_instances = math.floor(train_split * num_instances)
    test_instances = math.floor(test_split * num_instances)

    counter = 0
    for instance_index, instance_name in enumerate(os.listdir(old_dir)):
        if instance_name[-4:] != '.jpg' or instance_name[-7:] == 'nir.jpg':
            print(f'{instance_name} is not an RGB jpg!')
            continue
        rgb_src = f'{old_dir}{instance_name}'
        nir_src = f'{old_dir}{instance_name}'.replace('.jpg', '_nir.jpg')

        if counter < train_instances:
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            shutil.copy(rgb_src, train_path)
            shutil.copy(nir_src, train_path)
        elif counter < train_instances + test_instances:
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            shutil.copy(rgb_src, test_path)
            shutil.copy(nir_src, test_path)
        else:
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            shutil.copy(rgb_src, val_path)
            shutil.copy(nir_src, val_path)

        counter += 1



if __name__ == '__main__':
    if sys.platform == 'win32':
        config_path = ''
    elif sys.platform == 'darwin':
        config_path = '/Users/loganfrank/Desktop/code/mlcv-final-project/code/config/logan_mac.yaml'
    elif sys.platform == 'linux':
        config_path = ''

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
    
    # combine(image_directory)
    for subset in ['max', 'multi_class', 'single_class']:
        path = f'{image_directory}{subset}/'
        old_path = f'{path}old/'
        train_path = f'{path}train/'
        test_path = f'{path}test/'
        val_path = f'{path}val/'
        split_data(old_path, train_path, test_path, val_path)
    

    path = f'{image_directory}unsupervised/'
    old_path = f'{path}old/'
    train_path = f'{path}train/'
    test_path = f'{path}test/'
    val_path = f'{path}val/'
    split_unsupervised(old_path, train_path, test_path, val_path)
    

    