import math
import numpy as voNP
from sklearn.utils import voShuffle
from TcNeuronLayer import TcNeuronLayer
from TcTypeGradDesc import TcTypeGradDesc
from TcTypeActivation import TcTypeActivation

# Neural Network
class TcNeuralNetwork( ) :
    def __init__( aorSelf, aoShape, aiCountLayer, adDropOut = 0.2, aeActHidden = TcTypeActivation.XeSigmoid, aeActLast = TcTypeActivation.XeSigmoid ) :
        # Record number of inputs and hidden layers
        aorSelf.viCountInput = aiCountInput
        aorSelf.viCountLayer = aiCountLayer
        aorSelf.veActHidden  = aeActHidden
        aorSelf.veActLast    = aeActLast

        # Initialize Layers
        aorSelf.voLayer = [ ] # Initialize layers to empty list
        for kiI in range( len( aiCountLayer ) ) :
            # Handle First Layer
            if( kiI == 0 ) :
                aorSelf.voLayer.append( TcNeuronLayer( ( aiCountLayer[ kiI ], aoShape[ 0 ] ), 0.1, False, adDropOut, aeActHidden ) )
            # Handle Last Layer
            elif( kiI == len( aiCountLayer ) - 1 ) :
                aorSelf.voLayer.append( TcNeuronLater( ( aoShape[ 1 ], aiCountLayer[ kiI - 1 ] ), 0.1, False, adDropOut, aeActLast ) )
            # Handle Hidden Layers
            else:
                aorSelf.voLayer.append( TcNeuronLayer( ( aiCountLayer[ kiI ], aiCountLayer[ kiI - 1 ] ), 0.1, False, adDropOut, aeActLast ) )            

    def MForwardPass( aorSelf, adX ) :
        # Run Forward Pass on first layer
        kdA = aorSelf.voLayer[ 0 ].MForwardPass( adX )

        # Run Forward Pass on Hidden Layers and Last Layer
        for kiI in range( 1, len( aorSelf.voLayer ) ) :
            kdA = aorSelf.voLayer[ kiI ].MForwardPass( kdA )

        # Return result
        return( kdA )

    def MTrain( adX, adY, aorSelf, aiEpochs, adLR, adLambda, aeGradDesc, aiBatchSize ) :
        # Obtain the index of the last layer
        kiLast = len( aorSelf.voLayer ) - 1    

        for kiEpoch in range( aiEpochs ) :
            # Zero out the loss
            kdLoss = 0.0
            
            # Shuffle the input/output pairs
            kdX, kdY = voShuffle( adX, adY, random_state=0 )

            for kiI in range( kdX.shape[ 0 ] ) :
                # Execute a Foward Pass
                kdA = aorSelf.MForwardPass( kdX[ kiI ] )

                # Calculate the Loss
                kdLoss += aorSelf.MLoss( kdA, kdY[ kiI ] )

                # Calculate Deltas and Gradients
                aorSelf.MCalculateDeltaGrads( kdX[ kiI ], kdY[ kiI ] )

    def MLoss( aorSelf, adA, adY ) :   
        # if the last layer activation function is SoftMax
        if( aorSelf.veActLast == TcTypeActivation.XeSOFTMAX ) :
            kdLoss = -( adY * voNP.log( adA + 0.01 ) ).sum( )
        # Else handle all other activation functions
        else :
            kdDelta = ( adA - adY )
            kdLoss = ( kdDelta * kdDelta ).sum( )
        return( kdLoss )

    def MCalculateDeltaGrads( adX, adY ) :
        # Obtain the index of the last layer
        kiLayer = len( aorSelf.voLayer ) - 1

        # Calculate Deltas and Gradients for all layers
        while( kiLayer >= 0 ) :
            # Handle Last Layer
            if( kiLayer == ( len( aorSelf.voLayer ) - 1 ) ) :
                if( aorSelf.veActLast == TcTypeActivation.XeSOFTMAX ) :
                    aorSelf.voLayer[ kiLayer ].vdD = -adY + aorSelf.voLayer[ kiI ].vdA
                else :
                    aorSelf.voLayer[ kiLayer ].vdD = -( adY - aorSelf.voLayer[ kiI ] ) * aorSelf.voLayer[ kiI ].vdAd
            else :
                aorSelf.voLayer[ kiLayer ].vdD = voNP.dot( aorSelf.voLayer[ kiLayer + 1 ].vdW.T, aorSelf.voLayer[ kiLayer + 1 ].voD ) * aorSelf.voLayer[ kiLayer ].voAd

            # Determine the input
            if( kiLayer > 0 ) :
                kdInput = aorSelf.voLayyer[ kiLast - 1 ].vdA    # Input is the result of previous layer
            else : # Layer == 0
                kdInput = adX

            # Calculate Gradients
            aorSelf.voLayer[ kiLayer ].vdWg += voNP.dot( aorSelf.voLayer[ kiLayer ].vdD, kdInput.T )
            aorSelf.voLayer[ kiLayer ].vdBg += aorSelf.voLayer[ kiLayer ].vdD

            # Update Layer Index
            kiLayer = kiLayer - 1

   