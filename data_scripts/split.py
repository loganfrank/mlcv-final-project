## Basic Python libraries
import sys
import os
import shutil
import argparse
import yaml
import math
import glob
sys.path.append(os.getcwd() + '/')

from utils.functions import listdir

def rename(directory):
    for crop in os.listdir(directory):
        # Get the new path
        crop_path = f'{directory}{crop}/'

        # Skip over files (we only want directories at this point)
        if not os.path.isdir(crop_path):
            print(f'{crop_path} is not a directory!')
            continue

        for crop_class in os.listdir(crop_path):
            # Get the new path
            crop_class_path = f'{crop_path}{crop_class}/'

            # Skip over files (we only want directories at this point)
            if not os.path.isdir(crop_class_path):
                print(f'{crop_class_path} is not a directory!')
                continue

            for num, instance in enumerate(os.listdir(crop_class_path)):
                # Skip over the annoying files
                if instance[-4:] != '.jpg' and instance[-4:] != '.JPG' and instance[-5:] != '.jpeg' and instance[-5:] != '.JPEG':
                    print(f'{instance} is not a .jpg file')
                    os.remove(f'{crop_class_path}{instance}')
                    continue

                src_path = os.path.abspath(f'{crop_class_path}{instance}')
                dest_path = os.path.abspath(f'{crop_class_path}{crop}_{crop_class}_{num}.jpg')
                os.rename(src_path, dest_path)

def split(old_dir, train_dir, test_dir, val_dir):
    # Percent of total instances reserved for train and test, the remaining will go into val
    train_split = 0.7
    test_split = 0.1

    for crop in os.listdir(old_dir):
        # Get the new path
        crop_path = f'{old_dir}{crop}/'

        # Skip over files (we only want directories at this point)
        if not os.path.isdir(crop_path):
            print(f'{crop_path} is not a directory!')
            continue

        for crop_class in os.listdir(crop_path):
            # Get the paths for the old location and the new train, test, and val paths
            # If the new train, test, val paths don't exist, create them
            crop_class_path = os.path.abspath(f'{crop_path}/{crop_class}/')

            # Skip over files (we only want directories at this point)
            if not os.path.isdir(crop_class_path):
                print(f'{crop_class_path} is not a directory!')
                continue

            crop_class_train_path = os.path.abspath(f'{train_dir}{crop}/{crop_class}/')
            if not os.path.exists(crop_class_train_path):
                os.makedirs(crop_class_train_path)
            crop_class_test_path = os.path.abspath(f'{test_dir}{crop}/{crop_class}/')
            if not os.path.exists(crop_class_test_path):
                os.makedirs(crop_class_test_path)
            crop_class_val_path = os.path.abspath(f'{val_dir}{crop}/{crop_class}/')
            if not os.path.exists(crop_class_val_path):
                os.makedirs(crop_class_val_path)
            
            # Get the list of all images, the number of all images, and the values for the split
            list_imgs = os.listdir(crop_class_path)
            list_imgs = [im for im in list_imgs if im[-4:] == '.jpg']
            num_imgs = len(list_imgs)
            num_train = math.floor(train_split * num_imgs)
            num_test = math.floor(test_split * num_imgs)

            for num, img in enumerate(list_imgs):
                # Get the original path of the instance
                src_path = os.path.abspath(f'{crop_class_path}/{img}')

                # Move to the respective new train, test, or val path depending on the current iteration number
                if num < num_train:
                    dest_path = os.path.abspath(f'{crop_class_train_path}')
                    shutil.move(src_path, dest_path)
                elif num < num_train + num_test:
                    dest_path = os.path.abspath(f'{crop_class_test_path}')
                    shutil.move(src_path, dest_path)
                else:
                    dest_path = os.path.abspath(f'{crop_class_val_path}')
                    shutil.move(src_path, dest_path)

if __name__ == '__main__':
    if sys.platform == 'win32':
        config_path = '/Users/frank.580/Desktop/code/cse-fabe/config/logan_pc.yaml'
    elif sys.platform == 'darwin':
        config_path = '/Users/loganfrank/Desktop/research/agriculture/code/cse-fabe/config/logan_mac.yaml'
    elif sys.platform == 'linux':
        pass
    dataset = input('Please enter the crop you want to use: ')
    dataset_image_directory = f'{dataset}_image_directory'
    dataset_network_directory = f'{dataset}_network_directory'
    dataset_data_directory = f'{dataset}_data_directory'
    dataset_results_directory = f'{dataset}_results_directory'

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

    # Create the image save paths
    old_dir = f'{image_directory}old/'
    train_dir = f'{image_directory}train/'
    test_dir = f'{image_directory}test/'
    val_dir = f'{image_directory}val/'

    if not os.path.exists(f'{image_directory}train/'):
        os.makedirs(f'{image_directory}train/')
        os.makedirs(f'{image_directory}test/')
        os.makedirs(f'{image_directory}val/')
    
    rename(old_dir)
    split(old_dir, train_dir, test_dir, val_dir)

    