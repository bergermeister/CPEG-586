import sys
import numpy as voNP
from TcNeuronLayer import TcNeuronLayer

# Neural Network
class TcNeuralNetwork( ) :
    def __init__( aorSelf ) :
        # Initialize Layers
        aorSelf.voL1 = TcNeuronLayer( ( 100, 748 ) )
        aorSelf.voL2 = TcNeuronLayer( (  10, 100 ) )

    # Train the Neural Network
    def MTrain( aorSelf, adX, adY, aiEpochs ) :
        for kiEpoch in range( aiEpochs ) :
            for kiInput in range( len( adX ) ) :
                kdA1 = aorSelf.voL1.MForwardPass( adX[ kiInput ] )
                kdA2 = aorSelf.voL2.MForwardPass( kdA1 )
                kdD2 = -voNP.multiply( voNP.multiply( adY[ kiInput ] - kdA2, kdA2 ), ( 1 - kdA2 ) )
                kdD1 = voNP.multiply( voNP.dot( aorSelf.voL2.vdW.T, kdD2 ), kdA1 * ( 1 - kdA1 ) )

    def MForwardPass( aorSelf, adX ) :
        kdA1 = aorSelf.voL1.MForwardPass( adX )
        kdA2 = aorSelf.voL2.MForwardPass( kdA1 )
        return( kdA2 )

    def MLoss( aorSelf, adA, adY ) :
        kdDelta = adY - adA
        return( 0.5 * voNP.multiply( kdDelta, kdDelta ).Sum( ) )
   