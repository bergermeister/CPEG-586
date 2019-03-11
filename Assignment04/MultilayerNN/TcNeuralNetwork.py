import math
import numpy as voNP
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

    # Train the Neural Network using the Stochastic Gradient Descent (SGD) algorithm
    def MTrainSGD( aorSelf, adX, adY, aiEpochs ) :
        kdL = voNP.zeros( ( aiEpochs, 1 ), dtype='float64' )
        for kiEpoch in range( aiEpochs ) :
            kdL[ kiEpoch ] = 0
            for kiX in range( adX.shape[ 0 ] ) :
                # Forward Pass
                kdA1 = aorSelf.voL1.MForwardPass( adX[ kiX ] )
                kdA2 = aorSelf.voL2.MForwardPass( kdA1 )
    
                # Calculate Loss
                kdL[ kiEpoch ] += ( 0.5 * ( ( kdA2 - adY[ kiX ] ) * ( kdA2 - adY[ kiX ] ) ) ).sum( )

                # Calculate Deltas
                #kdD2 = -voNP.multiply( adY[ kiX ] - kdA2, kdA2 * ( 1 - kdA2 ) )
                #kdD1 = voNP.multiply( voNP.dot( aorSelf.voL2.vdW.T, kdD2 ), kdA1 * ( 1 - kdA1 ) )
                kdD2 = -voNP.multiply( adY[ kiX ] - kdA2, aorSelf.MDAct( kdA2 )  )
                kdD1 = voNP.multiply( voNP.dot( aorSelf.voL2.vdW.T, kdD2 ), aorSelf.MDAct( kdA1 ) )

                # Back Propagate
                aorSelf.voL1.MBackpropagate( kdD1 * adX[ kiX ].T, kdD1 )
                aorSelf.voL2.MBackpropagate( kdD2 * kdA1.T, kdD2 )

        return( kdL )

    # Train the Neural Network using the Mini-Batch Stocahstic Gradient Descent (SGD) algorithm
    def MTrainMB( aorSelf, adX, adY, aiEpochs, aiBatch ) :
        kdL = voNP.zeros( ( aiEpochs, 1 ), dtype='float64' )
        for kiEpoch in range( 0, aiEpochs ) :
            kdL[ kiEpoch ] = 0
            for kiX in range( 0, adX.shape[ 0 ], aiBatch ) :
                kdBatchX1 = voNP.zeros( ( adX.shape[ 1 ], adX.shape[ 2 ] ), dtype='float64' )
                kdBatchD1 = voNP.zeros( ( aorSelf.voL1.vdW.shape[ 0 ], 1 ), dtype='float64' )              
                kdBatchX2 = voNP.zeros( ( aorSelf.voL1.vdW.shape[ 0 ], 1 ), dtype='float64' )
                kdBatchD2 = voNP.zeros( ( aorSelf.voL2.vdW.shape[ 0 ], 1 ), dtype='float64' )     
                
                for kiBatch in range( kiX, kiX + aiBatch ) :
                    # Forward Pass
                    kdA1 = aorSelf.voL1.MForwardPass( adX[ kiBatch ] )
                    kdA2 = aorSelf.voL2.MForwardPass( kdA1 )
    
                    # Calculate Loss
                    kdL[ kiEpoch ] += ( 0.5 * ( ( kdA2 - adY[ kiBatch ] ) * ( kdA2 - adY[ kiBatch ] ) ) ).sum( )

                    # Calculate Deltas
                    #kdD2 = -voNP.multiply( adY[ kiBatch ] - kdA2, kdA2 * ( 1 - kdA2 ) )
                    #kdD1 = voNP.multiply( voNP.dot( aorSelf.voL2.vdW.T, kdD2 ), kdA1 * ( 1 - kdA1 ) )
                    kdD2 = -voNP.multiply( adY[ kiBatch ] - kdA2, aorSelf.MDAct( kdA2 )  )
                    kdD1 = voNP.multiply( voNP.dot( aorSelf.voL2.vdW.T, kdD2 ), aorSelf.MDAct( kdA1 ) )

                    # Update Batches
                    kdBatchX1 += adX[ kiBatch ] / float(aiBatch)
                    kdBatchD1 += kdD1 / float(aiBatch)
                    kdBatchX2 += kdA1 / float(aiBatch)
                    kdBatchD2 += kdD2 / float(aiBatch)

                # Back Propagate
                aorSelf.voL1.MBackpropagate( kdBatchD1 * kdBatchX1.T, kdBatchD1 )
                aorSelf.voL2.MBackpropagate( kdBatchD2 * kdBatchX2.T, kdBatchD2 )

        return( kdL )

    def MDAct( aorSelf, adA ) :
        # Tan Hyperbolic
        if( aorSelf.viAct == 1 ) :
            kdOut = ( 1 - ( adA * adA ) )
    
        # Rectified Linear Unit
        elif( aorSelf.viAct == 2 ) :
            kdOut = adA
            for kiIdx in range( len( adA ) ) :
                if( adA[ kiIdx ] > 0 ) :
                    kdOut[ kiIdx ] = 1
                else :
                    kdOut[ kiIdx ] = 0
    
        # Sigmoid
        else :
            kdOut = ( adA * ( 1 - adA ) )
        return( kdOut )
            

    def MForwardPass( aorSelf, adX ) :
        # Run Forward Pass on first layer
        kdA = aorSelf.voLayer[ 0 ].MForwardPass( adX )

        # Run Forward Pass on Hidden Layers and Last Layer
        for kiI in range( 1, len( aorSelf.voLayer ) ) :
            kdA = aorSelf.voLayer[ kiI ].MForwardPass( kdA )

        # Return result
        return( kdA )

    def MLoss( aorSelf, adA, adY ) :
        kdDelta = adY - adA
        return( 0.5 * voNP.multiply( kdDelta, kdDelta ).Sum( ) )
   