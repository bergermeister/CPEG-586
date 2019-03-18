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
        # If not using Mini Batch, force batch size to 1
        if( aeGradDesc != TcTypeGradDesc.XeMiniBatch ) :
            aiBatchSize = 1

        for kiEpoch in range( aiEpochs ) :
            # Zero out the loss
            kdLoss = 0.0
            
            # Shuffle the input/output pairs
            kdX, kdY = voShuffle( adX, adY, random_state = 0 )

            for kiI in range( 0, kdX.shape[ 0 ], aiBatchSize ) :
                # Get the Input and Expected Output Batches
                kdXb = kdX[ kiI : ( kiI + aiBatchSize ) ]
                kdYb = kdY[ kiI : ( kiI + aiBatchSize ) ]

                # Execute a Foward Pass
                kdA = aorSelf.MForwardPass( kdXb )

                # Calculate the Loss
                kdLoss += aorSelf.MLoss( kdA, kdYb )

                # If Gradient Descent is Stochastic or MiniBatch
                if( ( aeGradDesc == TcTypeGradDesc.XeStochastic ) or ( aeGradDesc ==  TcTypeGradDesc.XeMiniBatch ) ):
                    aorSelf.MBackPropagate( kdXb, kdYb, adLR )
            if( aeGradDesc == TcTypeGradDesc.XeBatch ) :
                aorSelf.MBackPropagate( kdXb, kdYb, adLR )
        return( kdLoss )

    def MLoss( aorSelf, adA, adY ) :   
        kiN = len( adA )
        kdL = 0.0
        for kiI in range( kiN ) :
            # if the last layer activation function is SoftMax
            if( aorSelf.veActLast == TcTypeActivation.XeSoftMax ) :
                kdL += -( adY[ kiI ] * voNP.log( adA[ kiI ] + 0.01 ) ).sum( )
            # Else handle all other activation functions
            else :
                kdDelta = ( adA[ kiI ] - adY[ kiI ] )
                kdL += ( kdDelta * kdDelta ).sum( )
        return( kdL )

    def MBackPropagate( aorSelf, adX, adY, adLR ) :
        kiN      = len( adX )                   # Obtain Batch Size
        kiLayers = len( aorSelf.voLayer )       # Obtain number of layers

        # For each element in the batch
        for kiI in range( kiN ) :
            # For each Layer in the Network, starting with the last layer
            for kiL in range( kiLayers - 1, -1, -1 ) :
                # Handle Last Layer
                if( kiL == ( kiLayers - 1 ) ) :
                    if( aorSelf.veActLast == TcTypeActivation.XeSoftMax ) :
                        kdD = aorSelf.voLayer[ kiL ].vdA[ kiI ] - adY[ kiI ]
                    else :
                        kdD = -( aorSelf.voLayer[ kiL ].vdA[ kiI ] - adY[ kiI ] ) * aorSelf.voLayer[ kiL ].vdAd[ kiI ]
                # Handle Intermediate Layers
                else :
                    kdD = voNP.dot( aorSelf.voLayer[ kiL + 1 ].vdW.T, kdD ) * aorSelf.voLayer[ kiL ].vdAd[ kiI ]

                    # Update Scale And Offset Gradients
                    aorSelf.voLayer[ kiL ].vdKg += ( kdD * aorSelf.voLayer[ kiL ].vdSh[ kiI ] ) / kiN
                    aorSelf.voLayer[ kiL ].vdOg += kdD / kiN

                    # Apply Batch Normalization 
                    kdD *= ( aorSelf.voLayer[ kiL ].vdK )
                    kdD /= ( float( kiN ) * ( ( ( aorSelf.voLayer[ kiL ].vdV + 1e-8 ) ** 0.5 ) ) )
                    kdD *= ( float( kiN - 1 ) - ( aorSelf.voLayer[ kiL ].vdSh[ kiI ] ** 2 ) ) 

                # Determine the input
                if( kiL > 0 ) :
                    kdInput = aorSelf.voLayer[ kiL - 1 ].vdA[ kiI ]    # Input is the result of previous layer
                else : # Layer == 0
                    kdInput = adX[ kiI ]

                # Calculate Gradients
                aorSelf.voLayer[ kiL ].vdWg += voNP.dot( kdD, kdInput.T )
                aorSelf.voLayer[ kiL ].vdBg += kdD

        # Backpropagate each layer
        for kiL in range( len( aorSelf.voLayer ) ) :
            aorSelf.voLayer[ kiL ].MBackpropagate( adLR, kiN )
            
