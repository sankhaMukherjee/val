from logs import logDecorator as lD

import json
import numpy      as np
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.AElib'

class AE():

    @lD.log(logBase + '.AE.__init__')
    def __init__(logger, self, nInp, layers, activations, nLatent, L=1.5):

        self.nInp    = nInp
        self.nLatent = nLatent
        self.L       = L

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

        # Here, we want to separate the encoder and the decoder.
        # So, for training, we shall generate placeholders that
        # can be filled with values of mu and sigma calculated
        # previously. 
        # ----------- Generate Placeholers for Mu and Sigma -----------
        self.LMu = tf.placeholder( dtype=tf.float32, 
                        shape=(None, nLatent))

        self.LSigma = tf.placeholder( dtype=tf.float32, 
                        shape=(None, nLatent))

        # Tensroflow needs a fixed size for random normal variables
        # So we shall use placeholders to pass in random normal
        # distributions into our decoder. Here, self.Latent is
        # simply a set of random normal distributions
        # ------------------------------------------------------------
        self.Latent = tf.placeholder( dtype=tf.float32, 
                        shape=(None, nLatent))

        self.decoder = self.Latent * self.LSigma + self.LMu 
        for i, (l, a) in enumerate(zip(
                        reversed(layers), reversed(activations))):
            self.decoder = tf.layers.dense(self.decoder, l, activation=a)

        self.decoder = tf.layers.dense(self.decoder, nInp, activation=tf.nn.sigmoid)

        # ------------------------------------------------------------
        # Generate the the cost functions
        # ------------------------------------------------------------
        self.aeErr = self.Inp * tf.log( self.decoder ) + (1-self.Inp) * tf.log( 1 - self.decoder )
        self.aeErr = tf.reduce_sum( self.aeErr, 1)
        self.aeErr = tf.reduce_mean( -1*self.aeErr )


        self.KLErr = tf.reduce_sum( self.sigma**2 + self.mu**2 - 1 - tf.log( self.sigma**2 ) , 1)
        self.KLErr = tf.reduce_mean( self.KLErr * 0.5 )


        self.Err   = (self.aeErr + self.L * self.KLErr)
        # ------------------------------------------------------------
        # Generate other misc operations
        # ------------------------------------------------------------
        self.Opt  = tf.train.AdamOptimizer().minimize( self.Err )
        self.init = tf.global_variables_initializer()
        
        return

