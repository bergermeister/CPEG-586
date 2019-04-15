import os
import sys
import math
import cv2
import numpy as voNP
import matplotlib.pyplot as voPlot
from multiprocessing import Pool, freeze_support, cpu_count
from TePool import TePool
from TeActivation import TeActivation
from TcLayer import TcLayer
from TcLayerC import TcLayerC
from TcCNNDeep import TcCNNDeep
from TcMatrix import TcMatrix

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

def MIsMatch( aoArgs ) :
   koCNN  = aoArgs[ 0 ]
   koPath = aoArgs[ 1 ]
   koFile = aoArgs[ 2 ]

   kdX = cv2.imread( koPath + '{0}'.format( koFile ), 0 ) / 255.0
   kdY = voNP.zeros( ( 10, 1 ) )
   kiY = int( koFile[ 0 ] )
   koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
   koX.vdData = kdX

   kdRes = koCNN.MForwardPass( koX )
   return( max( kdRes ) == kdRes[ kiY ] )

def MComputeAccuracy( aoCNN, aoPath ) :
   freeze_support( )
   kiCount = cpu_count( )
   koPool = Pool( kiCount )
   koFiles = [ ]

   # Read all file names
   for koFilename in os.listdir( aoPath ) :
      koFiles.append( koFilename )

   kdTotal    = len( koFiles )
   kdAccuracy = 0.0

   for kiI in range( 0, kdTotal, kiCount ) :
      koArgs = [ ( aoCNN, aoPath, koFiles[ kiB ] ) for kiB in range( kiCount ) ]
      koRes = koPool.map( MIsMatch, koArgs )
      kdAccuracy += voNP.sum( koRes )

   return( ( kdAccuracy / kdTotal ) * 100.0 )

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
   koCNNLayers.append( TcLayerC( ( kiCountFML1, 1 ), ( 28, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( kiCountFML2, kiCountFML1 ), ( 12, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )

   # Create a list of NN Layers. The second CNN layer produces an output of 4x4 per Feature Map
   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 50, 4 * 4 * kiCountFML2 ), TeActivation.XeRELU, 0.8 ) )
   koNNLayers.append( TcLayer( ( 10, 50 ), TeActivation.XeSoftMax, 1 ) )

   # Read MNist Training Data Set
   kdTrainX, kdTrainY = MReadMNIST( koMNIST + 'Training1000/' )

   # Create Deep CNN
   koCNN = TcCNNDeep( koCNNLayers, koNNLayers )

   # Train the CNN
   koCNN.MTrain( kdTrainX, kdTrainY, 30, 0.1, kiSizeBatch )

   # Test the CNN
   kdAccuracy = MComputeAccuracy( koCNN, koMNIST + 'Test10000/' )

   # Print Result Accuracy
   print( "Accuracy: ", kdAccuracy, "%")

if __name__ == "__main__" :
    main( )
