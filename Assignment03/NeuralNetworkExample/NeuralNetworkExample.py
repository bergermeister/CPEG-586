import os
import sys
import cv2
import matplotlib.pyplot as voPlot
#from sklearn.utils import shuffle
import numpy as voNP
from TcNeuralNetwork import TcNeuralNetwork

def MRunNeuralNetworkSGD( aiNeurons, aiActivation, aiEpochs, adTrainX, aiTrainY, adTestX, aiTestY ) :
    # Create a Neural Network with aiNeurons in Hidden Layer
    koNN = TcNeuralNetwork( aiNeurons, aiActivation )

    # Train the Neural Network
    koNN.MTrainSGD( adTrainX, aiTrainY, aiEpochs )

    # Test the Neural Network
    kdAccuracy = MRunNeuralNetwork( koNN, adTestX, aiTestY )

    # Return accuracy
    return( kdAccuracy )

def MRunNeuralNetworkMB( aiNeurons, aiActivation, aiEpochs, adTrainX, aiTrainY, adTestX, aiTestY ) :
    # Create a Neural Network with aiNeurons in Hidden Layer
    koNN = TcNeuralNetwork( aiNeurons, aiActivation )

    # Train the Neural Network
    koNN.MTrainMB( adTrainX, aiTrainY, aiEpochs, 10 )

    # Test the Neural Network
    kdAccuracy = MRunNeuralNetwork( koNN, adTestX, aiTestY )

    # Return accuracy
    return( kdAccuracy )

def MRunNeuralNetwork( aoNN, adTestX, aiTestY ) :
    # Record Accuracy
    kdAccuracy = 0

    # Test the Neural Network
    for kiI in range( aiTestY.shape[ 0 ] ) :
        # Forward Pass
        kiY = aoNN.MForwardPass( adTestX[ kiI ] )

        # Determine Index
        aiIndex = kiY.argmax( axis = 0 )
        if( aiTestY[ kiI, aiIndex ] == 1 ) :
            kdAccuracy = kdAccuracy + 1

    print( "Accuracy = ", kdAccuracy / 10000.0 )

    # Return accuracy
    return( kdAccuracy / 10000.0 )

def main( ) :
    # Create storage for training and test data
    kdTrain  = voNP.empty( (  1000, 28, 28 ), dtype='float64' )
    kiTrainY = voNP.zeros( (  1000, 10,  1 ) )
    kdTestX  = voNP.empty( ( 10000, 28, 28 ), dtype='float64' )
    kiTestY  = voNP.zeros( ( 10000, 10,  1 ) )

    # Load the images
    kiIdx = 0
    for koFilename in os.listdir( '../Resources/Data/Training1000/' ) :
        kiY = int( koFilename[ 0 ] )
        kiTrainY[ kiIdx, kiY ] = 1.0
        kdTrain [ kiIdx ] = cv2.imread( '../Resources/Data/Training1000/{0}'.format( koFilename ), 0 ) / 255.0
        kiIdx = kiIdx + 1

    # Load Test Data
    kiIdx = 0
    for koFilename in os.listdir( '../Resources/Data/Test10000' ) :
        kiY = int( koFilename[ 0 ] )
        kiTestY[ kiIdx, kiY ] = 1.0
        kdTestX[ kiIdx ] = cv2.imread( '../Resources/Data/Test10000/{0}'.format( koFilename ), 0 ) / 255.0
        kiIdx = kiIdx + 1

    kdTrainX = kdTrain.reshape( kdTrain.shape[ 0 ], kdTrain.shape[ 1 ] * kdTrain.shape[ 2 ], 1 )
    kdTestX  = kdTestX.reshape ( kdTestX.shape[ 0 ],  kdTestX.shape[ 1 ]  * kdTestX.shape[ 2 ], 1 )

    # Run the SGD Neural Networks
    kiValuesNeuron = [ 25, 50, 100, 150 ]
    kdAccuracy = voNP.empty( ( 3, 2, 4, 6 ), dtype = 'float64' )
    for kiAct in range( 1 ) :
        for kiMethod in range( 2 ) :
            for kiNeurons in range( 4 ) : # len( kiValuesNeuron ) ) :
                koLabel = "|A:" + str( kiAct ) + "|M:" + str( kiMethod ) + "|N:" + str( kiValuesNeuron[ kiNeurons ] )
                print( koLabel )

                # Create a Neural Network
                koNN = TcNeuralNetwork( kiValuesNeuron[ kiNeurons ], kiAct )
                for kiEpochs in range( 6 ) :
                    # Train the Network using either SGD or MB
                    if( kiMethod == 0 ) :
                        koNN.MTrainSGD( kdTrainX, kiTrainY, 25 )
                    else :
                        koNN.MTrainMB( kdTrainX, kiTrainY, 25, 10 )

                    # Record the Accuracy
                    kdAccuracy[ kiAct ][ kiMethod ][ kiNeurons ][ kiEpochs ] = MRunNeuralNetwork( koNN, kdTestX, kiTestY )
            
                # Plot the Accuracy trend
                voPlot.plot( range( 6 ), kdAccuracy[ kiAct ][ kiMethod ][ kiNeurons ], linestyle='-', marker='o', label=koLabel )

    voPlot.title( "Assignment 03 Plot" )
    voPlot.xlabel( "Epochs" ) 
    voPlot.ylabel( "Accuracy" )
    voPlot.legend( loc = 'upper left' )
    voPlot.show( )

if __name__ == "__main__" :
    main( )
