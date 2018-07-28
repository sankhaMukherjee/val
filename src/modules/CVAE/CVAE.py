import json, os
import numpy      as np
import tensorflow as tf

from lib  import CAElib 
from logs import logDecorator as lD 
from tqdm import tqdm 

from datetime import datetime as dt

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.CVAE.CVAE'


@lD.log(logBase + '.testVAE1')
def testCVAE(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    folder     = '../data/raw/mnist'
    file       = 't10k-images-idx3-ubytenpy.npy'
    labelsFile = 't10k-labels-idx1-ubytenpy.npy'
    
    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    print(X.shape, X.max(), X.min())

    # The labels need to be one hot encoded
    labels = np.load(os.path.join(folder, labelsFile))
    print(labels.shape, labels.max(), labels.min())
    mapper = np.eye(10)

    labelsOHE = np.array([mapper[l] for l in labels]).astype(np.float32)

    nInp, nLatent, nLabel, L = 784, 2, 10, 1
    layers           = [700, 500, 100]
    activations      = [tf.tanh, tf.tanh, tf.tanh]

    cvae = CAElib.CVAE(nInp, nLabel, layers, activations, nLatent, L)
    
    print('VAE instance generated')

    print('Fitting a function ...')
    cvae.fit(X, labelsOHE, 1000)
    
    print('Starting the save images ...')
    now        = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    imgFolder  = '../results/img/{}'.format(now)
    os.makedirs(imgFolder)

    print('Getting the latent state values')
    mu, sigma = cvae.getLatent(X, labelsOHE, cvae.restorePoints[-1])
    print(mu.mean(axis=0))

    print('Making predictions ...')
    xHat = cvae.predict(X[:20], labelsOHE[:20], cvae.restorePoints[-1])


    print('Plotting the latent space')
    
    print('Plotting some simple images')
    plt.figure(figsize=(10, 5))
    ax1 = plt.axes([0,0,0.5,1])
    ax2 = plt.axes([0.5,0,0.5,1])
    for i in range(xHat.shape[0]):

        ax1.cla(); ax2.cla()
        ax1.imshow( X[i].reshape(28, 28),    cmap=plt.cm.gray )
        ax2.imshow( xHat[i].reshape(28, 28), cmap=plt.cm.gray )
        ax1.set_xticks([]); ax1.set_yticks([]);
        ax2.set_xticks([]); ax2.set_yticks([]);
        plt.savefig(os.path.join(imgFolder, '{:05}.png'.format(i)))

    return


@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    testCVAE()

    return

