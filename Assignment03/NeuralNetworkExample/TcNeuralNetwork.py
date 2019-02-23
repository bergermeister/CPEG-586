import sys
import numpy as voNP
from TcNeuronLayer import TcNeuronLayer

# Neural Network
class TcNeuralNetwork( ) :
    def __init__( aorSelf, aiNeurons, aiActivation ) :
        # Initialize Layers
        aorSelf.voL1 = TcNeuronLayer( ( aiNeurons, 784 ), 0.1, aiActivation )
        aorSelf.voL2 = TcNeuronLayer( (  10, aiNeurons ), 0.1, aiActivation )
        aorSelf.viAct = aiActivation

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
        kdA1 = aorSelf.voL1.MForwardPass( adX )
        kdA2 = aorSelf.voL2.MForwardPass( kdA1 )
        return( kdA2 )

    def MLoss( aorSelf, adA, adY ) :
        kdDelta = adY - adA
        return( 0.5 * voNP.multiply( kdDelta, kdDelta ).Sum( ) )
   