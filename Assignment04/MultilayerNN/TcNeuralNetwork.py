import math
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from TcNeuronLayer import TcNeuronLayer
from TcTypeGradDesc import TcTypeGradDesc
from TcTypeActivation import TcTypeActivation

# Neural Network
class TcNeuralNetwork( ) :
    def __init__( aorSelf, aoShape, aiCountLayer, aeActHidden = TcTypeActivation.XeSigmoid, aeActLast = TcTypeActivation.XeSigmoid ) :
        # Record number of inputs and hidden layers
        aorSelf.veActHidden  = aeActHidden
        aorSelf.veActLast    = aeActLast

        # Initialize Layers
        aorSelf.voLayer = [ ] # Initialize layers to empty list
        for kiI in range( len( aiCountLayer ) ) :
            # Handle First Layer
            if( kiI == 0 ) :
                aorSelf.voLayer.append( TcNeuronLayer( ( aiCountLayer[ kiI ], aoShape[ 0 ] ), False, aeActHidden ) )
            # Handle Last Layer
            elif( kiI == len( aiCountLayer ) - 1 ) :
                aorSelf.voLayer.append( TcNeuronLayer( ( aoShape[ 1 ], aiCountLayer[ kiI - 1 ] ), False, aeActLast ) )
            # Handle Hidden Layers
            else:
                aorSelf.voLayer.append( TcNeuronLayer( ( aiCountLayer[ kiI ], aiCountLayer[ kiI - 1 ] ), False, aeActHidden ) )            

    def MForwardPass( aorSelf, adX ) :
        # Run Forward Pass on first layer
        kdA = aorSelf.voLayer[ 0 ].MForwardPass( adX )

        # Run Forward Pass on Hidden Layers and Last Layer
        for kiI in range( 1, len( aorSelf.voLayer ) ) :
            kdA = aorSelf.voLayer[ kiI ].MForwardPass( kdA )

        # Return result
        return( kdA )

    def MTrain( aorSelf, adX, adY, aiEpochs, adLR, aeGradDesc, aiBatchSize ) :
        # Obtain the index of the last layer
        kiLast = len( aorSelf.voLayer ) - 1    

        for kiEpoch in range( aiEpochs ) :
            # Zero out the loss
            kdLoss = 0.0
            
            # Shuffle the input/output pairs
            # kdX, kdY = voShuffle( adX, adY, random_state=0 )
            kdX = adX
            kdY = adY

            for kiI in range( kdX.shape[ 0 ] ) :
                # Execute a Foward Pass
                kdA = aorSelf.MForwardPass( kdX[ kiI ] )

                # Calculate the Loss
                kdLoss += aorSelf.MLoss( kdA, kdY[ kiI ] )

                # Calculate Deltas and Gradients
                aorSelf.MCalculateDeltaGrads( kdX[ kiI ], kdY[ kiI ] )

                # If Gradient Descent is Stochastic
                if( aeGradDesc == TcTypeGradDesc.XeStochastic ) :
                    aorSelf.MBackPropagate( adLR, 1 )
                elif( aeGradDesc ==  TcTypeGradDesc.XeMiniBatch ) :
                    if( ( kiI % aiBatchSize ) == ( aiBatchSize - 1 ) ) :
                        aorSelf.MBatchNormalization( kdX[ ( kiI - aiBatchSize + 1 ) : kiI + 1 ], 1, 1 )
                        aorSelf.MBackPropagate( adLR, aiBatchSize )
            if( aeGradDesc == TcTypeGradDesc.XeBatch ) :
                aorSelf.MBackPropagate( adLR, aiBatchSize )

    def MLoss( aorSelf, adA, adY ) :   
        # if the last layer activation function is SoftMax
        if( aorSelf.veActLast == TcTypeActivation.XeSoftMax ) :
            kdLoss = -( adY * voNP.log( adA + 0.01 ) ).sum( )
        # Else handle all other activation functions
        else :
            kdDelta = ( adA - adY )
            kdLoss = ( kdDelta * kdDelta ).sum( )
        return( kdLoss )

    def MCalculateDeltaGrads( aorSelf, adX, adY ) :
        # Obtain the index of the last layer
        kiLayer = len( aorSelf.voLayer ) - 1

        # Calculate Deltas and Gradients for all layers
        while( kiLayer >= 0 ) :
            # Handle Last Layer
            if( kiLayer == ( len( aorSelf.voLayer ) - 1 ) ) :
                if( aorSelf.veActLast == TcTypeActivation.XeSoftMax ) :
                    aorSelf.voLayer[ kiLayer ].vdD = -adY + aorSelf.voLayer[ kiLayer ].vdA
                else :
                    aorSelf.voLayer[ kiLayer ].vdD = -( adY - aorSelf.voLayer[ kiLayer ].vdA ) * aorSelf.voLayer[ kiLayer ].vdAd
            # Handle Intermediate Layers
            else :
                aorSelf.voLayer[ kiLayer ].vdD = voNP.dot( aorSelf.voLayer[ kiLayer + 1 ].vdW.T, aorSelf.voLayer[ kiLayer + 1 ].vdD ) * aorSelf.voLayer[ kiLayer ].vdAd

            # Determine the input
            if( kiLayer > 0 ) :
                kdInput = aorSelf.voLayer[ kiLayer - 1 ].vdA    # Input is the result of previous layer
            else : # Layer == 0
                kdInput = adX

            # Calculate Gradients
            aorSelf.voLayer[ kiLayer ].vdWg += voNP.dot( aorSelf.voLayer[ kiLayer ].vdD, kdInput.T )
            aorSelf.voLayer[ kiLayer ].vdBg += aorSelf.voLayer[ kiLayer ].vdD

            # Update Layer Index
            kiLayer = kiLayer - 1

    def MBackPropagate( aorSelf, adLR, aiBatch ) :
        # Backpropagate each layer
        for kiLayer in range( len( aorSelf.voLayer ) ) :
            aorSelf.voLayer[ kiLayer ].MBackpropagate( adLR, aiBatch )

    def MBatchNormalization( aorSelf, adX, adY, adB ) :
         kdMean = adX.sum( axis = 0 ) / len( adX )
         kdVar  = ( ( adX - kdMean ) ** 2 ).sum( axis = 0 )
