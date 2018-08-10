import json, os
import numpy      as np
import tensorflow as tf

import matplotlib.pyplot as plt

from lib  import ConvVAELib 
from logs import logDecorator as lD 
from tqdm import tqdm 

from datetime import datetime as dt

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.ConvVAE.ConvVAE'

@lD.log(logBase + '.testVAE1')
def testConvVAE(logger):
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
    now        = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    imgFolder  = '../results/img/{}'.format(now)
    os.makedirs(imgFolder)

    nInpX, nInpY, nInpCh = 28, 28, 1
    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    X = X.reshape(-1, nInpX, nInpY, nInpCh)
    print(X.shape, X.max(), X.min())

    labels = np.load(os.path.join(folder, labelsFile))

    # Parameters for the network
    filters     = [1, 7, 10]
    kernels     = [3, 4, 5]
    strides     = [2, 2, 1]
    activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]

    nLatent     = 2

    cVAE = ConvVAELib.ConvVAE(nInpX, nInpY, nInpCh, 
                filters, kernels, strides, activations, 
                nLatent, L=1.5)

    # X1 = X[:77, :, :, :]
    # with tf.Session() as sess:
    #     sess.run( cVAE.init )
    #     encoder = sess.run( cVAE.encoder, feed_dict = {cVAE.Inp: X1})
    #     mu      = sess.run( cVAE.mu, feed_dict = {cVAE.Inp: X1})

    #     latent  = np.random.normal( 0, 1, (mu.shape) )

    #     Err     = sess.run( cVAE.Err, feed_dict = {
    #                     cVAE.Inp: X1, cVAE.Latent : latent})


    plt.figure(figsize=(5, 8))
    for i in tqdm(range(40)):
        x, y = i//8, i%8
        ax = plt.axes([ x*0.2, y*0.125 , 0.2, 0.125])
        ax.imshow(X[i].reshape(28,28), cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig('../results/img/conv/X.png')
    plt.close('all')

    for it in tqdm(range(1000)):

        if it > 0:
            cVAE.fit(X, Niter=10, restorePoint=cVAE.restorePoints[-1])
        else:
            cVAE.fit(X, Niter=10)

        Xhat  = cVAE.predict(X[:40,:,:, :], cVAE.restorePoints[-1])
        Xhat1 = cVAE.predict2D(X[:2,:,:, :], cVAE.restorePoints[-1])
        tqdm.write('{}'.format(Xhat1.shape))
        
        plt.figure(figsize=(5, 8))

        for i in tqdm(range(40)):
            x, y = i//8, i%8
            ax = plt.axes([ x*0.2, y*0.125 , 0.2, 0.125])
            ax.imshow(Xhat[i].reshape(28,28), cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.savefig('../results/img/conv/Xhat_{:04d}.png'.format(it))
        plt.close('all')

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

    testConvVAE()

    return

