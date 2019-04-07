import os
import sys
import math
import cv2
import numpy as voNP
import matplotlib.pyplot as voPlot
from TePool import TePool
from TeActivation import TeActivation
from TcLayer import TcLayer
from TcLayerC import TcLayerC
from TcCNNDeep import TcCNNDeep

def MReadMNIST( aoPath ) :
   # Create empty of images and expected output
   kdX = []
   kdY = []

    # Load the images
   kiI = 0
   for koFilename in os.listdir( aoPath ) :
      kdX.append( cv2.imread( aoPath + '{0}'.format( koFilename ), 0 ) / 255.0 )
      kdY.append( voNP.zeros( ( 10, 1 ) ) )
      kiY = int( koFilename[ 0 ] )     
      kdY[ kiI ][ kiY ] = 1.0
      kiI = kiI + 1

   return( kdX, kdY )

def main( ) :
   #Set location of MNIST data
   koMNIST = '../../MNIST/'

   # Settings for Deep Convolutional Neural Network (Accuracy should be ~92%)
   kiSizeBatch = 5
   kiSizeKernl = 5   # Size of the kernel
   kiCountFML1 = 6   # Feature Maps in first layer
   kiCountFML2 = 12  # Feature Maps in second layer

   # Create a list of CNN Layers
   koCNNLayers = [ ]
   koCNNLayers.append( TcLayerC( ( kiCountFML1, 1 ), ( 28, kiSizeKernl, kiSizeBatch ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( kiCountFML2, kiCountFML1 ), ( 12, kiSizeKernl, kiSizeBatch ), TePool.XeAvg, TeActivation.XeRELU ) )

   # Create a list of NN Layers. The second CNN layer produces an output of 4x4 per Feature Map
   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 50, 4 * 4 * kiCountFML2, kiSizeBatch ), TeActivation.XeRELU, False, 0.8, 0.8 ) )
   koNNLayers.append( TcLayer( ( 10, 50, kiSizeBatch ), TeActivation.XeSoftMax, False, 0.8, 0.8 ) )

   # Read MNist Training Data Set
   kdTrainX, kdTrainY = MReadMNIST( koMNIST + 'Training1000/' )

   # Create Deep CNN
   koCNN = TcCNNDeep( koCNNLayers, koNNLayers, kiSizeBatch )

   # Train the CNN
   koCNN.MTrain( kdTrainX, kdTrainY, 30, 0.1, kiSizeBatch )

if __name__ == "__main__" :
    main( )
