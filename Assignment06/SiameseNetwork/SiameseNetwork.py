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
      kiI = kiI + 1

   return( kdX, kdY )

def MIsMatchMNIST( aoArgs ) :
   koCNN  = aoArgs[ 0 ]
   koPath = aoArgs[ 1 ]
   koFile = aoArgs[ 2 ]

   kdX = cv2.imread( koPath + '{0}'.format( koFile ), 0 ) / 255.0
   kiY = int( koFile[ 0 ] )
   koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
   koX.vdData = kdX

   kdRes = koCNN.MNetworkClassifier( koX )
   return( max( kdRes ) == kdRes[ kiY ] )

def MIsMatchATT( aoArgs ) :
   koCNN  = aoArgs[ 0 ]
   koPath = aoArgs[ 1 ]
   koFile = aoArgs[ 2 ]

   kdX = cv2.imread( koPath + '{0}'.format( koFile ), 0 ) / 255.0
   kiY = int( koFile.split( '_' )[ 0 ][ 1: ] )
   koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
   koX.vdData = kdX

   kdRes = koCNN.MNetworkClassifier( koX )
   return( max( kdRes ) == kdRes[ kiY - 1 ] )

def MComputeAccuracyMNIST( aoCNN, aoPath ) :
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
      koArgs = [ ( aoCNN, aoPath, koFiles[ kiI + kiB ] ) for kiB in range( kiCount ) ]
      koRes = koPool.map( MIsMatchMNIST, koArgs )
      kdAccuracy += voNP.sum( koRes )

   return( ( kdAccuracy / kdTotal ) * 100.0 )

def MComputeAccuracyATT( aoCNN, aoPath ) :
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
      koArgs = [ ( aoCNN, aoPath, koFiles[ kiI + kiB ] ) for kiB in range( kiCount ) ]
      koRes = koPool.map( MIsMatchATT, koArgs )
      kdAccuracy += voNP.sum( koRes )

   return( ( kdAccuracy / kdTotal ) * 100.0 )

def MTrainAndTestMNIST( aiSizeBatch, aiSizeKernel ) :
   koDataset = '../../MNIST/'

   # Create a list of CNN Layers
   koCNNLayers = [ ]
   koCNNLayers.append( TcLayerC( (  6,  1 ), ( 28, 28, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( 12,  6 ), ( 12, 12, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )

   # Create a list of NN Layers.
   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 50, 192 ), TeActivation.XeRELU, 0.8 ) )
   koNNLayers.append( TcLayer( ( 10, 50 ), TeActivation.XeSoftMax, 1 ) )

   # Read MNIST Training Data Set
   kdTrainX, kdTrainY = MReadMNIST( koDataset + 'Training1000/' )

   # Create Deep CNN
   koCNN = TcCNNSiamese( koCNNLayers, koNNLayers )

   # Train the CNN
   koCNN.MTrainModel( kdTrainX, kdTrainY, 30, 0.1, aiSizeBatch )
   koCNN.MTrainClassifier( kdTrainX, kdTrainY, 50, 0.1, aiSizeBatch )

   # Test the CNN
   kdAccuracy = MComputeAccuracy( koCNN, koDataset + 'Test10000/' )

   # Print Result Accuracy
   print( "Accuracy: ", kdAccuracy, "%")

def MTrainAndTestATT( aiSizeBatch, aiSizeKernel ) :
   koDataset = '../../ATTFaceDataSet/'

   # Create a list of CNN Layers
   koCNNLayers = [ ]
   koCNNLayers.append( TcLayerC( (  6,  1 ), ( 92, 112, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( 12,  6 ), ( 44,  54, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( ( 32,  1 ), ( 92, 112, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( ( 64, 32 ), ( 44,  54, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )
   #koCNNLayers.append( TcLayerC( ( 24, 12 ), ( 20,  25, aiSizeKernel ), TePool.XeAvg, TeActivation.XeRELU ) )

   # Create a list of NN Layers.
   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 500, 6000 ), TeActivation.XeRELU, 0.8 ) )
   #koNNLayers.append( TcLayer( ( 500, 40000 ), TeActivation.XeRELU, 0.8 ) )
   koNNLayers.append( TcLayer( ( 40, 500 ), TeActivation.XeSoftMax, 1 ) )

   # Read ATT Training Data Set
   kdTrainX, kdTrainY = MReadATT( koDataset + 'Training/' )

   # Create Deep CNN
   koCNN = TcCNNSiamese( koCNNLayers, koNNLayers )

   # Train the CNN
   koCNN.MTrainModel( kdTrainX, kdTrainY, 30, 0.1, aiSizeBatch )
   koCNN.MTrainClassifier( kdTrainX, kdTrainY, 1000, 0.1, aiSizeBatch )

   # Test the CNN
   kdAccuracy = MComputeAccuracy( koCNN, koDataset + 'Testing/' )

   # Print Result Accuracy
   print( "Accuracy: ", kdAccuracy, "%")

def main( ) :
   MTrainAndTestMNIST( 5, 5 )
   #MTrainAndTestATT( 5, 5 )   

if __name__ == "__main__" :
    main( )
