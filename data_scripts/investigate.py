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

def split_move(old, train, test, val):
    # old /Users/loganfrank/Desktop/code/mlcv-final-project/data/images/single_class/
    train_split = 0.7
    test_split = 0.15
    for cl in os.listdir(old):
        cl_path = f'{old}/{cl}'
        if not os.path.isdir(cl_path):
            continue
        if cl == 'test' or cl == 'train' or cl == 'val':
            continue
        train_cl_path = f'{train}{cl}/'
        test_cl_path = f'{test}{cl}/'
        val_cl_path = f'{val}{cl}/'

        if not os.path.exists(train_cl_path):
            os.makedirs(train_cl_path)

        if not os.path.exists(test_cl_path):
            os.makedirs(test_cl_path)

        if not os.path.exists(val_cl_path):
            os.makedirs(val_cl_path)

        list_imgs = os.listdir(cl_path)
        list_imgs = [im for im in list_imgs if im[-4:] == '.jpg']
        num_imgs = len(list_imgs) // 2
        num_train = math.floor(train_split * num_imgs)
        num_test = math.floor(test_split * num_imgs)
        for num, instance in enumerate(os.listdir(cl_path)):
            if instance[-4:] != '.jpg' or 'nir' in instance:
                continue
            src_rgb = f'{cl_path}/{instance}'
            src_nir = src_rgb.replace('.jpg', '_nir.jpg')

            # Move to the respective new train, test, or val path depending on the current iteration number
            if num < num_train:
                dest = os.path.abspath(f'{train_cl_path}/')
                shutil.move(src_rgb, dest)
                shutil.move(src_nir, dest)
            elif num < num_train + num_test:
                dest = os.path.abspath(f'{test_cl_path}')
                shutil.move(src_rgb, dest)
                shutil.move(src_nir, dest)
            else:
                dest = os.path.abspath(f'{val_cl_path}')
                shutil.move(src_rgb, dest)
                shutil.move(src_nir, dest)

def split_unsupervised(old, train, test, val):
    # old /Users/loganfrank/Desktop/code/mlcv-final-project/data/images/single_class/
    train_split = 0.7
    test_split = 0.15
    list_imgs = os.listdir(old)
    list_imgs = [im for im in list_imgs if im[-4:] == '.jpg']
    num_imgs = len(list_imgs) // 2
    num_train = math.floor(train_split * num_imgs)
    num_test = math.floor(test_split * num_imgs)
    for num, instance in enumerate(os.listdir(old)):
        if instance[-4:] != '.jpg' or 'nir' in instance:
            continue
        src_rgb = f'{path}/{instance}'
        src_nir = src_rgb.replace('.jpg', '_nir.jpg')

        # Move to the respective new train, test, or val path depending on the current iteration number
        if num < num_train:
            dest = os.path.abspath(f'{train}/')
            shutil.move(src_rgb, dest)
            shutil.move(src_nir, dest)
        elif num < num_train + num_test:
            dest = os.path.abspath(f'{test}')
            shutil.move(src_rgb, dest)
            shutil.move(src_nir, dest)
        else:
            dest = os.path.abspath(f'{val}')
            shutil.move(src_rgb, dest)
            shutil.move(src_nir, dest)

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

def combine(imgs_path):
    iii = 1
    for phase in ['train', 'val']:
        phase_path = f'{imgs_path}{phase}'
        label_path = f'{phase_path}/labels'
        num_classes = 6
        grayscale_value = 255 // num_classes
        num_instances = len(os.listdir(f'{label_path}/{[x for x in os.listdir(label_path) if x != ".DS_Store"][0]}'))
        classes = [x for x in os.listdir(label_path) if x != ".DS_Store"]
        class_dict = {(i + 1) : c for i, c in enumerate(classes)}



        for path in [f'{label_path}/{x}' for x in os.listdir(label_path) if x != ".DS_Store"]:
            assert len(os.listdir(path)) == num_instances
        for instance_index, instance_name in enumerate(os.listdir(f'{label_path}/{[x for x in os.listdir(label_path) if x != ".DS_Store"][0]}')):
            if instance_name[-4:] != '.png':
                print(instance_name)
                continue
            matrix = np.ones((512, 512))
            max_count = 0
            for class_index, class_path in enumerate([f'{label_path}/{x}' for x in os.listdir(label_path) if x != ".DS_Store"]):
                # Skip over files (we only want directories at this point)
                if not os.path.isdir(class_path):
                    print(f'{class_path} is not a directory!')
                    continue

                instance_path = f'{class_path}/{instance_name}'

                img = plt.imread(instance_path)
                count = np.sum(img)
                if count > max_count:
                    max_count = count
                    max_label = class_path.split('/')[-1]
                matrix[img == 1] = (class_index + 1) * grayscale_value
            uniques, uniques_counts = np.unique(matrix, return_counts=True)
            uniques = uniques[1:]
            uniques_counts = uniques_counts[1:]

            # get rgb and nir images
            rgb = f'{phase_path}/images/rgb/{instance_name[:-4]}.jpg'
            nir = f'{phase_path}/images/nir/{instance_name[:-4]}.jpg'

            if(not os.path.exists(f'{imgs_path}all_counts/num{len(uniques)}')):
                os.makedirs(f'{imgs_path}all_counts/num{len(uniques)}')
            if(not os.path.exists(f'{imgs_path}all_max/{max_label}')):
                os.makedirs(f'{imgs_path}all_max/{max_label}')


            rgb_counts = f'{imgs_path}all_counts/num{len(uniques)}/{max_label}_{iii}.jpg'
            nir_counts = f'{imgs_path}all_counts/num{len(uniques)}/{max_label}_{iii}_nir.jpg'
            rgb_max = f'{imgs_path}all_max/{max_label}/{max_label}_{iii}.jpg'
            nir_max = f'{imgs_path}all_max/{max_label}/{max_label}_{iii}_nir.jpg'

            shutil.copy(rgb, rgb_counts)
            shutil.copy(rgb, rgb_max)
            shutil.copy(nir, nir_counts)
            shutil.copy(nir, nir_max)

            iii += 1

def combine_unsupervised(imgs_path):
    iii = 1
    for phase in ['test']:
        phase_path = f'{imgs_path}{phase}'
        images_path = f'{phase_path}/images/nir/'

        for instance_index, instance_name in enumerate(os.listdir(images_path)):
            if instance_name[-4:] != '.jpg':
                print(instance_name)
                continue

            # get rgb and nir images
            rgb = f'{phase_path}/images/rgb/{instance_name[:-4]}.jpg'
            nir = f'{phase_path}/images/nir/{instance_name[:-4]}.jpg'

            if(not os.path.exists(f'{imgs_path}unsupervised/')):
                os.makedirs(f'{imgs_path}unsupervised/')

            rgb_dest = f'{imgs_path}unsupervised/unsupervised_{iii}.jpg'
            nir_dest = f'{imgs_path}unsupervised/unsupervised_{iii}_nir.jpg'

            shutil.copy(rgb, rgb_dest)
            shutil.copy(nir, nir_dest)

            iii += 1




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
        if not os.path.exists(f'{path}/train/'):
            os.makedirs(f'{path}/train/')
        if not os.path.exists(f'{path}/test/'):
            os.makedirs(f'{path}/test/')
        if not os.path.exists(f'{path}/val/'):
            os.makedirs(f'{path}/val/')
        train_path = f'{path}/train/'
        test_path = f'{path}/test/'
        val_path = f'{path}/val/'
        split_move(path, train_path, test_path, val_path)
    

    path = f'{image_directory}unsupervised/'
    if not os.path.exists(f'{path}/train/'):
        os.makedirs(f'{path}/train/')
    if not os.path.exists(f'{path}/test/'):
        os.makedirs(f'{path}/test/')
    if not os.path.exists(f'{path}/val/'):
        os.makedirs(f'{path}/val/')
    train_path = f'{path}/train/'
    test_path = f'{path}/test/'
    val_path = f'{path}/val/'
    split_unsupervised(path, train_path, test_path, val_path)
    

    