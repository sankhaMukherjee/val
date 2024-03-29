# val

This is a repository that looks into several different types of autoencoders. 

Practically all of this is experimental code, so use it with caution. Most of the code is distributed over multiple branches. They have not been merged, and probably will never be. Currently, the different branches available are the following:

 - master: This is the main branch. A basic fully-connected variational autoencoder is available in this branch. Most branches use this as a baseline to start exploring a new type of encoder.
 - dev: This is supposed to be the dev branch. Most branches branch out from here 
 - conv: This is an autoencoder that uses convolutional neural network
 - cvae: This is labeled variational autoencoder. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for `*nix`-like systems. These have been tried on macOS Sierra (Version 10.12.6) before. 

1. Clone the program to your computer. 
2. type `make firstRun`. This should do the following
    2.1. generate a virtual environment in folder `env`
    2.2. install a number of packages
    2.3. generate a new `requirements.txt` file
    2.4. generate an initial git repository
3. change to the `src` folder
4. run the command `make run`. This should run the small test program
5. Generate your documentation folder by running `make doc`. 
6. Check whether all the tests pass with the command `make test`. This uses py.test for running tests. 


## Built With

 - Python 3.6

## Contributing

Please send in a pull request.

## Authors

Sankha Mukherjee - Initial work (2018)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/sankhaMukherjee/val/blob/master/LICENCE.txt) file for details

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.
 