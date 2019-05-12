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
from TcCNNSiamese import TcCNNSiamese
from TcMatrix import TcMatrix

def MReadATT( aoPath ) :
   # Create empty of images and expected output
   kdX = []
   kdY = []

    # Load the images
   kiI = 0
   for koFilename in os.listdir( aoPath ) :
      kdX.append( cv2.imread( aoPath + '{0}'.format( koFilename ), 0 ) / 255.0 )
      kdY.append( voNP.zeros( ( 40, 1 ) ) )
      kiY = int( koFilename.split( '_' )[ 0 ][ 1: ] )
      kdY[ kiI ][ kiY - 1 ] = 1.0
      #kdY.append( voNP.zeros( ( 10, 1 ) ) )
      #kiY = int( koFilename[ 1 ] )
      #kdY[ kiI ][ kiY ] = 1.0
      kiI = kiI + 1

   return( kdX, kdY )

def MIsMatch( aoArgs ) :
   koCNN  = aoArgs[ 0 ]
   koPath = aoArgs[ 1 ]
   koFile = aoArgs[ 2 ]
   kdY = voNP.zeros( ( 40, 1 ) )
   kiY = int( koFile.split( '_' )[ 0 ][ 1: ] )
   kdY[ kiI ][ kiY - 1 ] = 1.0
   #kdX = cv2.imread( koPath + '{0}'.format( koFile ), 0 ) / 255.0
   #kiY = int( koFile[ 1 ] )
   #kdY[ kiI ][ kiY ] = 1.0
   koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
   koX.vdData = kdX

   kdRes = koCNN.MNetworkClassifier( koX )
   return( max( kdRes ) == kdRes[ kiY - 1 ] )

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
   # Set location of ATT data
   koATT = '../../ATTFaceDataSet/'
   #koATT = '../../MNIST/'

   # Settings for Deep Convolutional Neural Network (Accuracy should be ~92%)
   kiSizeBatch = 5
   kiSizeKernl = 5   # Size of the kernel

   # Create a list of CNN Layers
   koCNNLayers = [ ]
   koCNNLayers.append( TcLayerC( ( 32,  1 ), ( 92, 112, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( 64, 32 ), ( 44,  54, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( ( 24, 12 ), ( 20,  25, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( (  6,  1 ), ( 28, 28, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( ( 12,  6 ), ( 12, 12, kiSizeKernl ), TePool.XeAvg, TeActivation.XeRELU ) )

   # Create a list of NN Layers. The second CNN layer produces an output of 4x4 per Feature Map
   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 500, 40000 ), TeActivation.XeRELU, 0.8 ) )
   koNNLayers.append( TcLayer( ( 40, 500 ), TeActivation.XeSoftMax, 1 ) )
   #koNNLayers.append( TcLayer( ( 50, 192 ), TeActivation.XeRELU, 0.8 ) )
   #koNNLayers.append( TcLayer( ( 10, 50 ), TeActivation.XeSoftMax, 1 ) )

   # Read MNist Training Data Set
   kdTrainX, kdTrainY = MReadATT( koATT + 'Training/' )
   #kdTrainX, kdTrainY = MReadATT( koATT + 'Training1000/' )

   # Create Deep CNN
   koCNN = TcCNNSiamese( koCNNLayers, koNNLayers )

   # Train the CNN
   koCNN.MTrainModel( kdTrainX, kdTrainY, 5, 0.1, kiSizeBatch )
   koCNN.MTrainClassifier( kdTrainX, kdTrainY, 5, 0.1, kiSizeBatch )

   # Test the CNN
   kdAccuracy = MComputeAccuracy( koCNN, koATT + 'Testing/' )
   #kdAccuracy = MComputeAccuracy( koCNN, koATT + 'Test10000/' )

   # Print Result Accuracy
   print( "Accuracy: ", kdAccuracy, "%")

if __name__ == "__main__" :
    main( )
