#!/bin/bash 

echo "Downloading the MNIST dataset from "
echo "http://yann.lecun.com/exdb/mnist/  "
echo "-----------------------------------"

if [ ! -d '../data'  ]; then
    mkdir '../data'
fi

if [ ! -d '../data/raw'  ]; then
    mkdir '../data/raw'
fi

if [ ! -d '../data/raw/mnist'  ]; then
    mkdir '../data/raw/mnist'
fi

if [ ! -d '../data/intermediate'  ]; then
    mkdir '../data/intermediate'
fi

if [ ! -d '../data/final'  ]; then
    mkdir '../data/final'
fi

echo "+----------------------------------+"
echo "| downloading files                |"
echo "+----------------------------------+"

cd '../data/raw/mnist'
wget 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
wget 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
wget 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
wget 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

echo "+----------------------------------+"
echo "| All files downloaded ...         |"
echo "| unzipping the files              |"
echo "+----------------------------------+"

gunzip *.gz

echo "| All files unzipped               |"
echo "+----------------------------------+"


cd '../../../src'

