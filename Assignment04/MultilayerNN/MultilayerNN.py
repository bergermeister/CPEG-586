import os
import sys
import math
import cv2
import numpy as voNP
import matplotlib.pyplot as voPlot
from TcTypeActivation import TcTypeActivation
from TcTypeGradDesc import TcTypeGradDesc
from TcNeuralNetwork import TcNeuralNetwork


def main( ) :
    # Create storage for training and test data
    kdTrainX = voNP.empty( (  1000, 28, 28 ), dtype='float64' )
    kiTrainY = voNP.zeros( (  1000, 10,  1 ) )
    kdTestX  = voNP.empty( ( 10000, 28, 28 ), dtype='float64' )
    kiTestY  = voNP.zeros( ( 10000, 10,  1 ) )

    # Load the images
    kiIdx = 0
    for koFilename in os.listdir( '../Data/Training1000/' ) :
        kiY = int( koFilename[ 0 ] )
        kiTrainY[ kiIdx, kiY ] = 1.0
        kdTrainX[ kiIdx ] = cv2.imread( '../Data/Training1000/{0}'.format( koFilename ), 0 ) / 255.0
        kiIdx = kiIdx + 1

    # Load Test Data
    kiIdx = 0
    for koFilename in os.listdir( '../Data/Test10000' ) :
        kiY = int( koFilename[ 0 ] )
        kiTestY[ kiIdx, kiY ] = 1.0
        kdTestX[ kiIdx ] = cv2.imread( '../Data/Test10000/{0}'.format( koFilename ), 0 ) / 255.0
        kiIdx = kiIdx + 1

    kdTrainX = kdTrainX.reshape( kdTrainX.shape[ 0 ], kdTrainX.shape[ 1 ] * kdTrainX.shape[ 2 ], 1 )
    kdTestX  = kdTestX.reshape ( kdTestX.shape[ 0 ],  kdTestX.shape[ 1 ]  * kdTestX.shape[ 2 ], 1 )

    kiCount = 30
    kdAccuracy = voNP.zeros( ( kiCount, 1), dtype = 'float64' )

    koShape  = [ kdTrainX.shape[ 1 ], kiTrainY.shape[ 1 ] ]
    koLayers = [ 50, 10 ]
    koNN = TcNeuralNetwork( koShape, koLayers, TcTypeActivation.XeSigmoid, TcTypeActivation.XeSoftMax )

    for kiLoop in range( kiCount ) :
        kdL = koNN.MTrain( kdTrainX, kiTrainY, 1, 0.1, TcTypeGradDesc.XeMiniBatch, 10 )
        print( "Loss = ", kdL )
        for kiI in range( kiTestY.shape[ 0 ] ) :
            kdY = koNN.MForwardPass( kdTestX[ kiI : kiI + 1 ] )
            kiMax = kdY.argmax( axis = 1 )
            if( kiTestY[ kiI, kiMax ] == 1 ) :
                kdAccuracy[ kiLoop ] += 1
        kdAccuracy[ kiLoop ] = kdAccuracy[ kiLoop ] / 10000.0

    voPlot.plot( kdAccuracy, linestyle='-', marker='o' )
    voPlot.title( "Assignment 04 Plot" )
    voPlot.xlabel( "Epochs" ) 
    voPlot.ylabel( "Accuracy" )
    # voPlot.legend( loc = 'upper left' )
    voPlot.show( )

if __name__ == "__main__" :
    main( )
