import sys
import numpy as voNP

# Neuron Layer Class
class TcNeuronLayer( ) :
    def __init__( aorSelf, aorShape ) :
        # Initialize matrix of weights and biases
        aorSelf.vdW = voNP.random.uniform( low=0.01, high=0.1, size=aorShape )
        aorSelf.vdB = voNP.random.uniform( low=0.01, high=0.1, size=( aorShape[ 0 ], 1 ) )

    def MForwardPass( aorSelf, adX ) :
        kdSum = aorSelf.MSummation( adX )
        kdOut = aorSelf.MActivation( kdSum )
        return( kdOut )
    
    def MSummation( aorSelf, adX ) :
        return( voNP.dot( aorSelf.vdW, adX ) + aorSelf.vdB )

    def MActivation( aorSelf, adActual ) :
        # Sigmoid Activation
        return( 1 / ( 1 + voNP.exp( -adActual ) ) )
