from tqdm     import tqdm
from logs     import logDecorator as lD
from datetime import datetime     as dt

import json, os
import numpy      as np
import tensorflow as tf


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.AElib'

class AE():

    @lD.log(logBase + '.AE.__init__')
    def __init__(logger, self, nInp, layers, activations, nLatent, L=1.5):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger : {logging.Logger}
                The logger function
            self {[type]} -- [description]
            nInp {[type]} -- [description]
            layers {[type]} -- [description]
            activations {[type]} -- [description]
            nLatent {[type]} -- [description]
        
        Keyword Arguments:
            L {number} -- [description] (default: {1.5})
        '''

        self.nInp    = nInp
        self.nLatent = nLatent
        self.L       = L
        self.restorePoints = []

        self.Inp  = tf.placeholder(
                        dtype=tf.float32, 
                        shape=(None, nInp))

        # ------------------------------------------------------------
        # Generate the encoder network
        # ------------------------------------------------------------
        self.encoder = None
        for i, (l, a) in enumerate(zip(layers, activations)):
            if i == 0:
                self.encoder = tf.layers.dense(self.Inp, l, activation=a)
            else:
                self.encoder = tf.layers.dense(self.encoder, l, activation=a)

        # -- Dense map to Latent Space ---------
        # Note the sigmas has to be positive. Hence the sigmoid activation
        self.mu    = tf.layers.dense(self.encoder, nLatent, activation=None)
        self.sigma = tf.layers.dense(self.encoder, nLatent, activation=tf.nn.sigmoid)

        # ------------------------------------------------------------
        # Generate the decoder network
        # ------------------------------------------------------------

        #Here, self.Latent is simply a set of random normal distributions
        # ------------------------------------------------------------
        self.Latent = tf.placeholder( dtype=tf.float32, 
                        shape=(None, nLatent))

        # self.decoder = self.Latent * self.LSigma + self.LMu 
        self.decoder = self.Latent * self.sigma + self.mu 
        for i, (l, a) in enumerate(zip(
                        reversed(layers), reversed(activations))):
            self.decoder = tf.layers.dense(self.decoder, l, activation=a)

        self.decoder = tf.layers.dense(self.decoder, nInp, activation=tf.nn.sigmoid)

        # ------------------------------------------------------------
        # Generate the the cost functions
        # ------------------------------------------------------------
        # ----- Reconstruction Error ----------------
        self.aeErr = self.Inp * tf.log( self.decoder ) + (1-self.Inp) * tf.log( 1 - self.decoder )
        self.aeErr = tf.reduce_sum( self.aeErr, 1)
        self.aeErr = tf.reduce_mean( -1*self.aeErr )

        # ----- KL Divergence-------------------------
        self.KLErr = tf.reduce_sum( self.sigma**2 + self.mu**2 - 1 - tf.log( self.sigma**2 ), 1)
        self.KLErr = tf.reduce_mean( self.KLErr * 0.5 )

        # ----- Total Error-------------------------
        self.Err   = (self.aeErr + self.L * self.KLErr)

        # ------------------------------------------------------------
        # Generate other misc operations
        # ------------------------------------------------------------
        self.Opt  = tf.train.AdamOptimizer().minimize( self.Err )
        self.init = tf.global_variables_initializer()
        
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

        return

    @lD.log(logBase + '.AE.saveModel')
    def saveModel(logger, self, sess):
        '''[summary]
        
        [description]
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            sess {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        '''

        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        modelFolder = '../models/{}'.format(now)
        os.makedirs(modelFolder)

        path = self.saver.save( sess, os.path.join( modelFolder, 'model.ckpt' ) )
        self.restorePoints.append(path)

        return path

    @lD.log(logBase + '.AE.getLatent')
    def getLatent(logger, self, X, restorePoint=None):
        try:
            with tf.Session() as sess:
                sess.run(self.init)

                # Try to restore an older checkpoint
                # ---------------------------------------
                if restorePoint is not None:
                    try:
                        self.saver.restore(sess, restorePoint)
                    except Exception as e:
                        logger.error('Unable to restore the session at [{}]:{}'.format(
                            restorePoint, str(e)))


                mu, sigma = sess.run([self.mu, self.sigma], 
                                feed_dict = { self.Inp : X} )

                return mu, sigma   
        except Exception as e:
            logger.error('Unable to fit the model: {}'.format( str(e) ))

        return

    @lD.log(logBase + '.AE.fit')
    def fit(logger, self, X, Niter=101, restorePoint=None):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        
        Keyword Arguments:
            Niter {number} -- [description] (default: {101})
            restorePoint {[type]} -- [description] (default: {None})
        '''

        try:
            with tf.Session() as sess:
                sess.run(self.init)

                # Try to restore an older checkpoint
                # ---------------------------------------
                if restorePoint is not None:
                    try:
                        self.saver.restore(sess, restorePoint)
                    except Exception as e:
                        logger.error('Unable to restore the session at [{}]:{}'.format(
                            restorePoint, str(e)))


                mu, sigma = sess.run([self.mu, self.sigma], 
                                feed_dict = { self.Inp : X} )

                print('Initial Latent State mean:')
                print(mu.mean(axis=0))

                for i in tqdm(range(Niter)):

                    latent = np.random.normal( 0, 1, mu.shape )

                    _, aeError, KLErr, Err = sess.run(
                            [self.Opt, self.aeErr, self.KLErr, self.Err], 
                            feed_dict = {
                                self.Inp    : X,
                                self.Latent : latent})


                mu, sigma = sess.run([self.mu, self.sigma], 
                                feed_dict = { self.Inp : X} )
                print('Final Latent State mean:')
                print(mu.mean(axis=0))

                self.saveModel(sess)
        except Exception as e:
            logger.error('Unable to fit the model: {}'.format( str(e) ))

        return

    @lD.log(logBase + '.AE.predict')
    def predict(logger, self, X, restorePoint=None):
        try:
            with tf.Session() as sess:
                sess.run(self.init)

                # Try to restore an older checkpoint
                # ---------------------------------------
                if restorePoint is not None:
                    try:
                        self.saver.restore(sess, restorePoint)
                    except Exception as e:
                        logger.error('Unable to restore the session at [{}]:{}'.format(
                            restorePoint, str(e)))

                mu     = sess.run(self.mu, feed_dict = {self.Inp : X})
                latent = np.random.normal( 0, 1, mu.shape )
                xHat   = sess.run(self.decoder, 
                        feed_dict = {
                            self.Inp    : X,
                            self.Latent : latent })

                return xHat
                
        except Exception as e:
            logger.error('Unable to fit the model: {}'.format( str(e) ))

        return