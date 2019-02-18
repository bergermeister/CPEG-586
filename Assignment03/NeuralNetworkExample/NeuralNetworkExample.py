import sys
import numpy as voNP
from TcNeuralNetwork import TcNeuralNetwork

def main( ) :
    koNN = TcNeuralNetwork( )
    kdInput = voNP.random.uniform( low=0.0, high=1000.0, size=( 1, 748, 1 ) )
    kdOutput = voNP.random.uniform( low=0.0, high= 1000.0, size=( 1, 10, 1 ) )
    koNN.MTrain( kdInput, kdOutput, 1 )
    #koNN.MForwardPass( kdInput )
    print( koNN.voL1.vdW.shape )
    print( koNN.voL1.vdB.shape )
    print( koNN.voL2.vdW.shape )
    print( koNN.voL2.vdB.shape )

if __name__ == "__main__" :
    main( )
