import sys
import numpy as voNP
from Neuron2 import GTcNeuron2

def GMProblem3( ) :
    aoNeuron = GTcNeuron2( )
    GMProblem3PartATrain( aoNeuron )
    GMProblem3PartBTest( aoNeuron )

def GMProblem3PartATrain( aorNeuron ) :
    # Open the Training Data File
    koFile = open( "Training.txt", "r" )

    # Instantiate empty lists for training inputs and outputs
    kdTrainIn  = []
    kdTrainOut = []

    # Read the Training Data from the file
    for koLine in koFile :
        koParts = koLine.split( ',' )
        kdTrainIn.append( [ float( koParts[ 0 ] ), float( koParts[ 1 ] ) ] )
        kdTrainOut.append( float( koParts[ 2 ] ) )

    # Train the Neuron
    print( "Problem 3 Part A - Training Neuron" )    
    aorNeuron.MTrain( kdTrainIn, kdTrainOut, 100000 )
    print( "Neuron Trained: W1 = ", aorNeuron.vdW[ 0 ], " W2 = ", aorNeuron.vdW[ 1 ], " Bias = ", aorNeuron.vdB )

def GMProblem3PartBTest( aorNeuron ) :
    # Open the Test Data FIle
    koFile = open( "TestData.txt", "r" )

    # Instantiate empty lists for Test Data Inputs and Outputs
    kdIn  = []
    kdOut = []

    # Read the Test Data from the file
    for koLine in koFile :
        koParts = koLine.split( ',' )
        kdIn.append( [ float( koParts[ 0 ] ), float( koParts[ 1 ] ) ] )
        kdOut.append( float( koParts[ 2 ] ) )

    # Test the Neuron
    for kiIdx in range( len( kdIn ) ) :
        kdAct = aorNeuron.MForwardPass( kdIn[ kiIdx ] )
        print( "Actual = ", kdAct, " Expected = ", kdOut[ kiIdx ] )
    
