## Basic Python libraries
import os
import sys
import glob
from collections import OrderedDict

def convert_data_parallel(network):
    """
    Convert a DataParallel model's weights to regular weights with ease
    """
    state_dict = OrderedDict()
    for k, v in network.state_dict().items():
        name = k[7:]
        state_dict[name] = v
    return state_dict

def listdir(current_directory, expr):
    """
    Used to replace os.listdir because it's trash
    """
    complete_path = f'{current_directory}{expr}'
    files = glob.glob(os.path.abspath(complete_path))
    file_names = []
    for index, name in enumerate(files):
        if sys.platform == 'win32':
            name = name.split('\\')[-1]
        elif sys.platform == 'darwin' or sys.platform == 'linux':
            name = name.split('/')[-1] 
        file_names.append(name)
    return file_names