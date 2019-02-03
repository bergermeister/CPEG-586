import sys
import numpy as voNP

# Neuron class with 2 inputs and a bias
class GTcNeuron2( ):
    def __init__( aorSelf ) :
        # Initialize array of weights and bias
        aorSelf.vdW = [ 0.1, 0.1 ]      # Initialize W1, W2
        aorSelf.vdB  = 0.1              # Initialize B
        aorSelf.vdLR = 0.01             # Initialize Learning Rate

    def MACtivation( aorSelf, adAct ) :
        # No activation function yet, simply bypass
        return( adAct )

    def MForwardPass( aorSelf, adX ) :
        # Apply the weights and call the activation function
        kdAct = voNP.dot( adX, aorSelf.vdW ) + aorSelf.vdB
        return( aorSelf.MACtivation( kdAct ) )

    def MLoss( aorSelf, adIn, adOut, adAct ) :
        # Loss = ( -x1 )( Ye - Ya ), 
        kdGradient = voNP.ndarray( ( 3, 1 ) )
        kdGradient[ 0 ] = ( -1 * ( adIn[ 0 ] ) * ( adOut - adAct ) )
        kdGradient[ 1 ] = ( -1 * ( adIn[ 1 ] ) * ( adOut - adAct ) )
        kdGradient[ 2 ] = ( -1 * (    1      ) * ( adOut - adAct ) )
        return( kdGradient )

    def MTrain( aorSelf, adIn, adOut, aiEpochs ) :
        for kiEpoch in range( aiEpochs ) :
            for kiIn in range( len( adIn ) ) :
                # Loop through training examples
                kdAct = aorSelf.MForwardPass( adIn[ kiIn ] )
            
                # Compute Loss
                kdGradient = aorSelf.MLoss( adIn[ kiIn ], adOut[ kiIn ], kdAct )

                # Back propagate
                aorSelf.vdW[ 0 ] = aorSelf.vdW[ 0 ] - ( aorSelf.vdLR * kdGradient[ 0 ][ 0 ] )
                aorSelf.vdW[ 1 ] = aorSelf.vdW[ 1 ] - ( aorSelf.vdLR * kdGradient[ 1 ][ 0 ] )
                aorSelf.vdB      = aorSelf.vdB      - ( aorSelf.vdLR * kdGradient[ 2 ][ 0 ] )

                if kiEpoch == ( aiEpochs - 1 ) :
                    print( "Actual = ", kdAct )