from logs import logDecorator as lD 
import json, os, struct
import numpy as np

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.mnistConvert.mnistConvert'


@lD.log(logBase + '.readLabel')
def readLabel(logger, fileName):
    '''[summary]
    
    [description]
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger object
    fileName : {str}
        path the to file containing binary data for the labels
        to be converted to a numpy array
    '''

    try:
        with open(fileName, 'rb') as f:
            magicNumber = struct.unpack('>i', f.read(4))[0]
            if magicNumber != 2049:
                logger.error('Unable to obtain the right magic number')
                return None

            N = struct.unpack('>i', f.read(4))[0]
            print('Number of items: {}'.format(N))

            data = struct.unpack('>{}B'.format(N), f.read(N))

            print('The first 10 lables: {}'.format(data[:10]))
            return np.array(data)

            
    except Exception as e:
        logger.error('Unable to read the file: [{}]'.format(fileName, str(e)))
        return None

    return

@lD.log(logBase + '.readData')
def readData(logger, fileName):
    '''[summary]
    
    [description]
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger object
    fileName : {str}
        path the to file containing binary data for the image data
        that needs to be converted.
    '''

    try:
        with open(fileName, 'rb') as f:
            magicNumber = struct.unpack('>i', f.read(4))[0]
            print('magic number = {}'.format(magicNumber))

            if magicNumber != 2051:
                logger.error('Unable to obtain the right magic number')
                return None

            N, x, y = struct.unpack('>3i', f.read(4*3))
            print('Number of items, shapes: {}, {}, {}'.format(N, x, y))

            data = struct.unpack('>{}B'.format(N*x*y), f.read(N*x*y))
            data = np.array(data)
            data = data.reshape(N,-1)

            return data

    except Exception as e:
        logger.error('Unable to read the file: [{}]'.format(fileName, str(e)))
        return None

    return


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    folder = '../data/raw/mnist'
    files = [f for f in os.listdir(folder) if not f.endswith('.npy')]

    labels = [os.path.join(folder, f) for f in files if '-labels-' in f]
    images = [os.path.join(folder, f) for f in files if '-images-' in f]

    for l in labels:
        data = readLabel(l)
        print(data.shape)
        print(data[:10])
        np.save(l+'npy', data)
        
    if not os.path.exists('../results/tmp'):
        os.makedirs('../results/tmp')

    for m, img in enumerate(images):
        data = readData(img)
        print(data.shape)
        print(data[:10])
        np.save(img+'npy', data)

        plt.figure(figsize=(10,1))
        for j in range(10):
            ax = plt.axes([j*0.1, 0, 0.1, 1])
            ax.imshow(data[j].reshape(28, 28), cmap=plt.cm.gray)

        plt.savefig('../results/tmp/{}.png'.format(m))
        
    print(files)

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

    doSomething()

    return

