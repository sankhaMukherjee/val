import json, os
import numpy      as np
import tensorflow as tf

from lib  import AElib 
from logs import logDecorator as lD 

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.module1.module1'


@lD.log(logBase + '.testVAE')
def testVAE(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    folder = '../data/raw/mnist'
    file   = 't10k-images-idx3-ubytenpy.npy'

    X = np.load(os.path.join(folder, file)).astype(np.float32)
    X = X /255.0
    print(X.shape, X.max(), X.min())

    nInp, nLatent, L = 784, 2, 1
    layers           = [700, 500, 10]
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
            vae.LMu    : mu,
            vae.LSigma : sigma,
            vae.Latent : latent })

        print(xHat.shape)

        aeError, KLErr, Err = sess.run([vae.aeErr, vae.KLErr, vae.Err], feed_dict = {
            vae.Inp    : X,
            vae.LMu    : mu,
            vae.LSigma : sigma,
            vae.Latent : latent})

        print('Autoencoder error: {}'.format(aeError))
        print('KL Divergence    : {}'.format(KLErr))
        print('Total error      : {}'.format(Err))

        for _ in range(10):

            latent = np.random.normal( 0, 1, mu.shape )
            sess.run(vae.Opt, feed_dict = {
                vae.Inp    : X,
                vae.LMu    : mu,
                vae.LSigma : sigma,
                vae.Latent : latent})
            
            aeError, KLErr, Err = sess.run([vae.aeErr, vae.KLErr, vae.Err], feed_dict = {
                vae.Inp    : X,
                vae.LMu    : mu,
                vae.LSigma : sigma,
                vae.Latent : latent})

            print('[{:10.3f}]-[{:10.3f}]-[{:10.3f}]'.format(aeError, KLErr, Err))
            
        


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

    testVAE()

    return

