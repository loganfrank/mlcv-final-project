## Basic Python imports
import os

## PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms

## Image / Array imports
import numpy as np
import pandas as pd
from PIL import Image

#### TEMP
import matplotlib.pyplot as plt 
import matplotlib as mpl

class balance_dataset(Dataset):
    
    def __init__(self, image_root_directory, dataframe, transform=None, phase='train', balance=False, cut=None, mode='rgb'):
        # The root directory containing all image data for a desired set/phase
        self.image_root_directory = os.path.abspath(f'{image_root_directory}{phase}/')
        dataframe_subset = dataframe.loc[phase]

        # Get the list of instance file names
        self.instances = dataframe_subset.index.get_level_values(1).to_numpy().astype('U')

        # Get the crop associated with each instance
        self.crops = dataframe_subset.index.get_level_values(0).to_numpy().astype('U')

        # Get the ground truth class labels for each instance, define the set of unique output classes and assign them an integer index
        self.classes = dataframe_subset.to_numpy().astype('U')
        self.classes = np.squeeze(self.classes)
        self.mode = mode
        self.classes_unique, classes_counts = np.unique(self.classes, return_counts=True)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index
        self.index_to_classes = {}
        for index, value in enumerate(self.classes_unique):
            self.index_to_classes[index] = value

        # Balance the dataset if flagged to do so
        if balance:

            # Get the number of instances in the class with the largest number of instances
            max_class = max(classes_counts)
            if cut is not None:
                cut = np.asscalar(sorted(classes_counts)[cut - 1])

            # Reset our class variables for the lists of instance image names, instance class labels, and instance crop labels
            self.instances = np.empty(0)
            self.classes = np.empty(0)
            self.crops = np.empty(0)

            # Create 2D matrices where each row is a class and each column is an instance of that class
            if cut is None:
                instances_2D = np.empty((0, max_class))
                labels_2D = np.empty((0, max_class))
                crops_2D = np.empty((0, max_class))
            elif isinstance(cut, int) and cut > 0:
                instances_2D = np.empty((0, cut))
                labels_2D = np.empty((0, cut))
                crops_2D = np.empty((0, cut))
            else:
                raise Exception('Invalid value for cut - must be either None or an int greater than 0.')

            # Loop through each possible class and perform the balancing computations
            for row_index, class_value in enumerate(self.classes_unique):
                # Get the instance image names, instance class labels, and instance crop labels for each class
                class_instances = dataframe_subset[dataframe_subset['anomaly'] == class_value].index.get_level_values(1).to_numpy().astype('U')
                class_labels = dataframe_subset[dataframe_subset['anomaly'] == class_value].to_numpy().astype('U')
                class_crops = dataframe_subset[dataframe_subset['anomaly'] == class_value].index.get_level_values(0).to_numpy().astype('U')
                
                # Create the array used for doing the same shuffle on all arrays
                shuffle = np.arange(class_instances.shape[0])
                np.random.shuffle(shuffle)

                # Shuffle the lists of instance image names, instance class labels, and instance crop labels
                class_instances = class_instances[shuffle]
                class_labels = class_labels[shuffle]
                class_crops = class_crops[shuffle]
                
                # Concatenate the arrays until the exceed the largest number of instances in a class
                while len(class_instances) < max_class:
                    class_instances = np.concatenate((class_instances, class_instances))
                    class_labels = np.concatenate((class_labels, class_labels))
                    class_crops = np.concatenate((class_crops, class_crops))

                # Crop the end so the class now has a number of instances equal to the largest number of instances in a class
                if cut is None:
                    class_instances = class_instances[:max_class]
                    class_labels = class_labels[:max_class]
                    class_crops = class_crops[:max_class]
                else:
                    class_instances = class_instances[:cut]
                    class_labels = class_labels[:cut]
                    class_crops = class_crops[:cut]

                # Concatenate / append to the rows of the 2D matrices
                instances_2D = np.concatenate((instances_2D, class_instances.reshape(1, -1)), axis=0)
                labels_2D = np.concatenate((labels_2D, class_labels.reshape(1, -1)), axis=0)
                crops_2D = np.concatenate((crops_2D, class_crops.reshape(1, -1)), axis=0)

            # Segment data into batches
            # Below is the number of instances per class in each batch (in this class batch size 30)
            num_per_class = 1
            for i in range(0, (len(class_instances) // num_per_class) * num_per_class, num_per_class):
                # Extract one batch worth of information
                batch_instances = instances_2D[:, i : i + num_per_class]
                batch_labels = labels_2D[:, i : i + num_per_class]
                batch_crops = crops_2D[:, i : i + num_per_class]

                # Add to global list
                self.instances = np.concatenate((self.instances, batch_instances.flatten().reshape(-1, 1)), axis=None)
                self.classes = np.concatenate((self.classes, batch_labels.flatten().reshape(-1, 1)), axis=None)
                self.crops = np.concatenate((self.crops, batch_crops.flatten().reshape(-1, 1)), axis=None)
                
            # If the length of any of the above global arrays is not equal to the number of instances in the largest
            # class times the number of classes, then more data remains to append to the global list
            if len(self.instances) < (len(class_instances) * len(self.classes_unique)):
                i += num_per_class
                batch_instances = instances_2D[:, i:]
                batch_labels = labels_2D[:, i:]
                batch_crops = crops_2D[:, i:]
                self.instances = np.concatenate((self.instances, batch_instances.flatten().reshape(-1, 1)), axis=None)
                self.classes = np.concatenate((self.classes, batch_labels.flatten().reshape(-1, 1)), axis=None)
                self.crops = np.concatenate((self.crops, batch_crops.flatten().reshape(-1, 1)), axis=None) 

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

        # Save the transform to apply to the data before loading
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Open the input image using PIL
        if self.mode == 'rgb':
            image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{self.instances[index]}'))
        if self.mode == 'nir':
            nir_instance = self.instances[index]
            nir_instance = nir_instance.replace('.jpg', '_nir.jpg')
            image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{nir_instance}'))
        if self.mode == 'rgbnir':
            rgb = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{self.instances[index]}'))
            nir = self.instances[index]
            nir = nir.replace('.jpg', '_nir.jpg')
            nir = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{nir}'))

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        # Perform the data transform on the image
        if self.transform and self.mode != 'rgbnir':
            image = self.transform(image)
        else:
            rgb, nir = self.transform(rgb, nir)
            image = torch.cat((rgb, nir), dim=0)

        return (image, class_index)


class imbalance_dataset(Dataset):
    
    def __init__(self, image_root_directory, dataframe, transform=None, phase='train', mode='rgb'):
        # The root directory containing all image data for a desired set/phase
        self.image_root_directory = os.path.abspath(f'{image_root_directory}{phase}/')
        dataframe_subset = dataframe.loc[phase]

        # Get the list of instance file names
        self.instances = dataframe_subset.index.get_level_values(1).to_numpy().astype('U')

        self.mode = mode 

        # Get the ground truth class labels for each instance, define the set of unique output classes and assign them an integer index
        self.classes = dataframe_subset.to_numpy().astype('U')
        self.classes = np.squeeze(self.classes)
        self.classes_unique = np.unique(self.classes)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index
        self.index_to_classes = {}
        for index, value in enumerate(self.classes_unique):
            self.index_to_classes[index] = value

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

        # Save the transform to apply to the data before loading
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Open the input image using PIL
        if self.mode == 'rgb':
            image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{self.instances[index]}'))
        if self.mode == 'nir':
            nir_instance = self.instances[index]
            nir_instance = nir_instance.replace('.jpg', '_nir.jpg')
            image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{nir_instance}'))
        if self.mode == 'rgbnir':
            rgb = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{self.instances[index]}'))
            nir = self.instances[index]
            nir = nir.replace('.jpg', '_nir.jpg')
            nir = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.classes[index]}/{nir}'))

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        # Perform the data transform on the image
        if self.transform and self.mode != 'rgbnir':
            image = self.transform(image)
        else:
            rgb, nir = self.transform(rgb, nir)
            image = torch.cat((rgb, nir), dim=0)

        return (image, class_index)

class synthetic_dataset(Dataset):
    
    def __init__(self, path_a, path_b, path_c):
        # Load the matrixes
        A = np.genfromtxt(fname=path_a, delimiter=',')
        B = np.genfromtxt(fname=path_b, delimiter=',')
        C = np.genfromtxt(fname=path_c, delimiter=',')

        # Initialize DataFrame with its index and column names
        index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['phase', 'class', 'instance'])
        dataframe = pd.DataFrame(index=index, columns=['softmaxA', 'softmaxB', 'softmaxC'])

        # Add all instances to the dataframe
        for index, instance in enumerate(A):
            dataframe.loc[('val', 'A', f'a{index}')] = instance

        for index, instance in enumerate(B):
            dataframe.loc[('val', 'B', f'b{index}')] = instance

        for index, instance in enumerate(C):
            dataframe.loc[('val', 'C', f'c{index}')] = instance

        # Was forced to have 3 values in MultiIndex, this removes the unnecessary val
        self.dataframe = dataframe.loc['val']

        # Get the instances
        self.instances = self.dataframe.index.get_level_values(1).to_numpy().astype('U')

        # Get the values
        self.values = self.dataframe.to_numpy()

        # Get the classes
        self.classes = self.dataframe.index.get_level_values(0).to_numpy().astype('U')
        self.classes_unique = np.unique(self.classes)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index
        self.index_to_classes = {}
        for index, value in enumerate(self.classes_unique):
            self.index_to_classes[index] = value

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Get the synthetic instance
        instance = self.values[index]

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        return (instance, class_index)



"""
class experiment4_dataset(Dataset):
    
    def __init__(self, image_root_directory, dataframe, transform=None, phase='train', num_smallest=None):
        # The root directory containing all image data for a desired set/phase
        self.image_root_directory = os.path.abspath(f'{image_root_directory}{phase}/')
        dataframe_subset = dataframe.loc[phase]

        # Get the list of instance file names
        self.instances = dataframe_subset.index.get_level_values(1).to_numpy().astype('U')

        # Get the crop associated with each instance
        self.crops = dataframe_subset.index.get_level_values(0).to_numpy().astype('U')

        # Get the ground truth class labels for each instance, define the set of unique output classes and assign them an integer index
        self.classes = dataframe_subset.to_numpy().astype('U')
        self.classes = np.squeeze(self.classes)
        self.classes_unique, classes_counts = np.unique(self.classes, return_counts=True)
        self.classes_to_index = {}
        for index, value in enumerate(self.classes_unique):
            self.classes_to_index[value] = index

        # Balance the dataset if flagged to do so
        if num_smallest is not None:
            # Reset our class variables for the lists of instance image names, instance class labels, and instance crop labels
            self.instances = np.empty(0)
            self.classes = np.empty(0)
            self.crops = np.empty(0)

            # Sort the classes by lowest number of instances to largest number of instances
            classes_counts_indexes = classes_counts.argsort()
            self.classes_unique = self.classes_unique[classes_counts_indexes]
            classes_counts = classes_counts[classes_counts_indexes]

            # Add the unbalanced classes to the global array of instances
            for i in range(num_smallest - 1):
                disease = self.classes_unique[i]
                class_instances = dataframe_subset[dataframe_subset['disease'] == disease].index.get_level_values(1).to_numpy().astype('U')
                class_labels = dataframe_subset[dataframe_subset['disease'] == disease].to_numpy().astype('U')
                class_crops = dataframe_subset[dataframe_subset['disease'] == disease].index.get_level_values(0).to_numpy().astype('U')

                self.instances = np.concatenate((self.instances, class_instances.reshape(-1, 1)), axis=None)
                self.classes = np.concatenate((self.classes, class_labels.reshape(-1, 1)), axis=None)
                self.crops = np.concatenate((self.crops, class_crops.reshape(-1, 1)), axis=None)
                
            # Create 2D matrices where each row is a class and each column is an instance of that class
            length = classes_counts[num_smallest - 1]
            instances_2D = np.empty((0, length))
            labels_2D = np.empty((0, length))
            crops_2D = np.empty((0, length))

            # Loop through each possible class and perform the balancing computations
            for row_index, class_value in enumerate(self.classes_unique[num_smallest - 1:]):
                # Get the instance image names, instance class labels, and instance crop labels for each class
                class_instances = dataframe_subset[dataframe_subset['disease'] == class_value].index.get_level_values(1).to_numpy().astype('U')
                class_labels = dataframe_subset[dataframe_subset['disease'] == class_value].to_numpy().astype('U')
                class_crops = dataframe_subset[dataframe_subset['disease'] == class_value].index.get_level_values(0).to_numpy().astype('U')
                
                # Create the array used for doing the same shuffle on all arrays
                shuffle = np.arange(class_instances.shape[0])
                np.random.shuffle(shuffle)

                # Shuffle the lists of instance image names, instance class labels, and instance crop labels
                class_instances = class_instances[shuffle]
                class_labels = class_labels[shuffle]
                class_crops = class_crops[shuffle]

                # Crop the end so the class now has a number of instances equal to the largest number of instances in a class
                class_instances = class_instances[:length]
                class_labels = class_labels[:length]
                class_crops = class_crops[:length]

                # Concatenate / append to the rows of the 2D matrices
                instances_2D = np.concatenate((instances_2D, class_instances.reshape(1, -1)), axis=0)
                labels_2D = np.concatenate((labels_2D, class_labels.reshape(1, -1)), axis=0)
                crops_2D = np.concatenate((crops_2D, class_crops.reshape(1, -1)), axis=0)
            
            # Add to global list
            self.instances = np.concatenate((self.instances, instances_2D.flatten().reshape(-1, 1)), axis=None)
            self.classes = np.concatenate((self.classes, labels_2D.flatten().reshape(-1, 1)), axis=None)
            self.crops = np.concatenate((self.crops, crops_2D.flatten().reshape(-1, 1)), axis=None)        
        else:
            raise Exception("num_smallest must be some integer 1 <= num_smallest < num_classes")

        # Map each instance's ground truth class label to its associated integer index
        self.classes_index = np.zeros((len(self.instances)))
        for index, class_name in enumerate(self.classes):
            self.classes_index[index] = self.classes_to_index[class_name]

        # Save the transform to apply to the data before loading
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Open the input image using PIL
        image = Image.open(os.path.abspath(f'{self.image_root_directory}/{self.crops[index]}/{self.classes[index]}/{self.instances[index]}'))

        # Identify the class label and convert it to long
        class_index = self.classes_index[index]
        class_index = class_index.astype(np.int64)

        # Perform the data transform on the image
        if self.transform:
            image = self.transform(image)

        # Visualize the post-transform image
        
        # Below three lines need to be commented out
        pil_trans = transforms.ToPILImage()
        img = pil_trans(image)
        img = np.array(img)

        plt.imshow(img)
        plt.show()

        return (image, class_index)
"""


