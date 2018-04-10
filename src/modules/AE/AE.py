import json, os
import numpy      as np
import tensorflow as tf

from lib  import AElib 
from logs import logDecorator as lD 
from tqdm import tqdm 

from datetime import datetime as dt

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.module1.module1'


@lD.log(logBase + '.testVAE')
def testVAE(logger):
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

    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    print(X.shape, X.max(), X.min())

    labels = np.load(os.path.join(folder, labelsFile))

    nInp, nLatent, L = 784, 2, 1
    layers           = [700, 500, 100]
    activations      = [tf.tanh, tf.tanh, tf.tanh]

    vae = AElib.AE(nInp, layers, activations, nLatent, L)
    print('VAE instance generated')

    with tf.Session() as sess:
        print('Initializing a session ...')
        sess.run(vae.init)

        print('Running the encoder ...')
        mu, sigma = sess.run([vae.mu, vae.sigma], 
            feed_dict = { vae.Inp : X} )

        print('Running the decoder ...')
        latent = np.random.normal( 0, 1, mu.shape )
        xHat   = sess.run(vae.decoder, feed_dict = {
            vae.Inp    : X,
            vae.Latent : latent })

        print(xHat.shape)

        aeError, KLErr, Err = sess.run([vae.aeErr, vae.KLErr, vae.Err], feed_dict = {
            vae.Inp    : X,
            vae.Latent : latent})

        print('Autoencoder error: {}'.format(aeError))
        print('KL Divergence    : {}'.format(KLErr))
        print('Total error      : {}'.format(Err))


        print('Starting an optimization run')
        for i in tqdm(range(5001)):

            latent = np.random.normal( 0, 1, mu.shape )
            _, aeError, KLErr, Err = sess.run([vae.Opt, vae.aeErr, vae.KLErr, vae.Err], feed_dict = {
                vae.Inp    : X,
                vae.Latent : latent})

            tqdm.write('AE error: [{:10.3f}] KL error: [{:10.3f}] Total Error: [{:10.3f}]'.format(aeError, KLErr, Err))
            
            if i < 10:
                saveCrit = 2 
            elif i < 100:
                saveCrit = 10
            elif i < 200:
                saveCrit = 20
            else:
                saveCrit = 50

            if i % saveCrit == 0:

                XHat = sess.run(vae.decoder, feed_dict = {
                    vae.Inp    : X,
                    vae.Latent : latent})            

                # plt.figure()
                # plt.scatter(mu[:, 0], mu[:, 1], c=labels)
                # plt.savefig('{}/mu_{:010}.png'.format(imgFolder, i))

                plt.figure( figsize=(10,6) )

                for j in range(10):
                    for k in range(3):

                        ax = plt.axes([j*0.1, + k*1.0/6, 0.1, 1.0/6])
                        ax.imshow( XHat[k*10+j].reshape(28, 28), cmap=plt.cm.hot )
                        ax.set_xticks([]); ax.set_yticks([]);

                        ax = plt.axes([j*0.1, 0.5 + k*1.0/6, 0.1, 1.0/6])
                        ax.imshow( X[k*10+j].reshape(28, 28), cmap=plt.cm.hot  )
                        ax.set_xticks([]); ax.set_yticks([]);

                plt.savefig('{}/comparison_{:010}.png'.format(imgFolder, i))

                plt.close('all')

        print('Saving the model')


        print('Running the decoder')

        mu, sigma = sess.run([vae.mu, vae.sigma], 
                feed_dict = { vae.Inp : X} )

        print('mu    = {} -> {}'.format(mu.max(axis=0), mu.min(axis=0)))
        print('sigma = {} -> {}'.format(sigma.max(axis=0), sigma.min(axis=0)))

        
        print('Generating the new set of images ...')
        print('------------------------------------')
        muMax     = mu.max(axis=0)
        muMin     = mu.min(axis=0)
        muMean    = list(mu.mean(axis=0))
        sigmaMean = list(mu.mean(axis=0))
        # Just look at the first two axes ...
        Nmax = 20
        muNew, sigmaNew = [], []
        for i in np.linspace(muMin[0], muMax[0], Nmax):
            for j in np.linspace(muMin[1], muMax[1], Nmax):
                temp = [i, j] + muMean[2:]
                muNew.append( temp )
                sigmaNew.append( sigmaMean )

        muNew    = np.array(muNew)
        sigmaNew = np.array(sigmaNew)
        latent   = np.random.normal( 0, 1, muNew.shape )

        newVal = sess.run( vae.decoder, feed_dict = {
                vae.mu    : muNew,
                vae.sigma : sigmaNew,
                vae.Latent : latent })

        plt.figure(figsize=(10,10))
        frac = 1/Nmax
        for i in tqdm(range(Nmax)):
            for j in tqdm(range(Nmax)):
                ax = plt.axes([i*frac, j*frac, frac, frac])
                ax.imshow( newVal[i*Nmax+j].reshape(28, 28), cmap=plt.cm.gray )
                ax.set_xticks([]); ax.set_yticks([]);

        plt.savefig('{}/generated.png'.format(imgFolder))
        plt.close('all')
            
    return

@lD.log(logBase + '.testVAE1')
def testVAE1(logger):
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

    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    print(X.shape, X.max(), X.min())

    labels = np.load(os.path.join(folder, labelsFile))

    nInp, nLatent, L = 784, 2, 1
    layers           = [700, 500, 100]
    activations      = [tf.tanh, tf.tanh, tf.tanh]

    vae = AElib.AE(nInp, layers, activations, nLatent, L)
    
    print('VAE instance generated')

    print('Fitting a function ...')
    vae.fit(X, 2000)
    
    print('Getting the latent state values')
    mu, sigma = vae.getLatent(X, vae.restorePoints[-1])
    print(mu.mean(axis=0))

    print('Making predictions ...')
    xHat = vae.predict(X[:20], vae.restorePoints[-1])

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

@lD.log(logBase + '.loadSavedModel')
def loadSavedModel(logger):
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
    
    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    print(X.shape, X.max(), X.min())

    labels = np.load(os.path.join(folder, labelsFile))
    distLabels = sorted(list(set(labels)))
    print(distLabels)

    nInp, nLatent, L = 784, 2, 1
    layers           = [700, 500, 100]
    activations      = [tf.tanh, tf.tanh, tf.tanh]

    vae = AElib.AE(nInp, layers, activations, nLatent, L)
    
    print('VAE instance generated')
    vae.restorePoints.append( '../models/2018-04-03--00-19-41/model.ckpt' )

    mu, sigma = vae.getLatent(X, vae.restorePoints[-1])

    plt.figure(figsize=(4,4))
    plt.axes([0.1, 0.1, 0.89, 0.89])
    for l in distLabels:
        rows = labels == l
        plt.scatter(mu[rows, 0], mu[rows, 1], c=plt.cm.tab10(l/10), label=str(l))
    plt.axis('equal')
    plt.legend()
    plt.savefig('../results/img/mu-{}.png'.format(dt.now().strftime('%Y-%m-%d--%H-%M-%S')))
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

    # testVAE()
    # testVAE1()
    loadSavedModel()

    return

