<h1>config:</h1>
This is where I keep all of my yaml config files. The config files are just paths to my data folders (images, where I save weights, results, dataframes, etc.)

<h1>data_scripts:</h1>
This is for all data parsing scripts, i.e. what I used (in conjunction with some manual work in the terminal) to move the images around and then to create the dataframe that I use in each of my training/testing scripts. There shouldn't be any needed to use any of these files, but they're here. I don't remember exactly but I think I used combine from investigate.py to combine the original data, then used move_data.py to split everything up.

<h1>networks:</h1>
These are the networks I have used. Each are modified versions of the ones from the torchvision package. The modifications are changing the ReLU to LeakyReLU and changing the input conv layer for the different number of channel experiments we are doing. Number of classes for output FC layer is determined with the constructor of the network object.

<h1>scripts:</h1>
Train and test scripts I used for training and testing my networks.

<h1>utils:</h1>
Contains other files I use in my pipeline. 
datasets.py -- creates the PyTorch Dataset object, contains objects for an imbalanced dataset and balanced (by replication) dataset, can create train/test/val sets from the input dataframe. I used the balanced one.

evaluate_network.py -- a helper function for evaluating a network on the validation set or test set, this is called at the end of the test script from the scripts directory.

functions.py -- random little functions, such as a listdir using glob and converting the weights of a DataParallel model to the weights of a single GPU model

parameters.py -- a small class that gets pickled to save the parameters used in each experiment

train_network.py -- a helper function for training a network, this is called at the end of the train script from the scripts directory.

transforms.py -- different PyTorch transforms I created