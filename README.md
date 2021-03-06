# pooch-detector

![pooch-detector in action](https://github.com/cptanalatriste/pooch-detector/blob/master/img/pooch_detected.png?raw=true)

A neural network for detecting dog breeds. 
It uses a transfer learning approach based on the [VGG-16 model 
architecture](https://arxiv.org/abs/1409.1556).


## Getting started
To train the network, be sure to do the following first:

1. Clone this repository.
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 
And the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz) if you want to apply the network to people.
3. Place the dataset files in your cloned copy of the repository.
4. Make sure you have installed all the Python packages defined in `requirements.txt`.

You can also [download the state dictionaries of the pre-trained networks](https://drive.google.com/open?id=16iCEDprKa0t4PfwaE50htfxEm9PY8tGv), if you prefer.

## Instructions
To explore the training process, you can take a look at the `dog_app.ipynb` jupyter notebook.
The network code is contained in the `pooch_detector` module.
