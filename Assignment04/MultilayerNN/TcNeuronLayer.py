import numpy as voNP
from TcTypeActivation import TcTypeActivation

# Neuron Layer Class
class TcNeuronLayer( object ) :
    def __init__( aorSelf, aorShape, abLast = False, aeActivation = TcTypeActivation.XeSigmoid ) :
        # Initialize weights, gradient of weights, biases, and gradient of biases
        aorSelf.vdW  = voNP.random.uniform( low=-0.1, high=0.1, size=aorShape )
        aorSelf.vdWg = voNP.zeros( aorShape )
        aorSelf.vdB  = voNP.random.uniform( low=-1, high=1, size=( aorShape[ 0 ], 1 ) )
        aorSelf.vdBg = voNP.zeros( ( aorShape[ 0 ], 1 ) )

        # Initialize Adaptive Moment Estimation (Adam) variables
        aorSelf.vdWm = voNP.zeros( aorShape )               # Past Gradient
        aorSelf.vdWv = voNP.zeros( aorShape )               # Past Squared Gradient
        aorSelf.vdBm = voNP.zeros( ( aorShape[ 0 ], 1 ) )   # Past Gradient
        aorSelf.vdBv = voNP.zeros( ( aorShape[ 0 ], 1 ) )   # Past Squared Gradient
        aorSelf.vdB1 = 0.9
        aorSelf.vdB2 = 0.999
        aorSelf.vdT  = 0

        # Store learning rate, last layer flag, drop out rate, and activation type
        aorSelf.vbLast    = abLast
        aorSelf.veAct     = aeActivation

        # Initialize Batch Normalization And Standard Neuron Layer Members
        aorSelf.vdScale  = 1.0                                  # Scale Factor  (Gamma)
        aorSelf.vdOffset = 0.0                                  # Offset Factor (Beta)
        aorSelf.vdM  = voNP.zeros( ( aorShape[ 0 ], 1 ) )       # Batch Mean
        aorSelf.vdV  = voNP.zeros( ( aorShape[ 0 ], 1 ) )       # Batch Variance
        aorSelf.vdS  = voNP.zeros( ( 1, aorShape[ 0 ], 1 ) )    # Batch of Sums               (S)
        aorSelf.vdSh = voNP.zeros( ( 1, aorShape[ 0 ], 1 ) )    # Mean Adjusted Batch Sum     (Sh)
        aorSelf.vdSb = voNP.zeros( ( 1, aorShape[ 0 ], 1 ) )    # Batch Normalized Output     (Sb)
        aorSelf.vdA  = voNP.zeros( ( 1, aorShape[ 0 ], 1 ) )    # Batch of Outputs            (A)
        aorSelf.vdAd = voNP.zeros( ( 1, aorShape[ 0 ], 1 ) )    # Batch of output derivations (Ad)

    def MForwardPass( aorSelf, adX ) :
        kiN = len( adX )                                        # Get size of Batch (N)
        kdS = voNP.zeros( ( kiN, aorSelf.vdW.shape[ 0 ], 1 ) )  # Initialize array of sums    (S)
        kdA = voNP.zeros( ( kiN, aorSelf.vdW.shape[ 0 ], 1 ) )  # Initialize array of outputs (A)

        # Calculate the Sum
        kdS = aorSelf.MSummation( adX )

        # If this is the last layer
        if( aorSelf.vbLast or kiN == 1 ) :
            # Last Layer does not have Batch Normalized Sums
            aorSelf.vdSh = None
            aorSelf.vdSb = None
            # Pass Sums to Activation Function
            kdA = aorSelf.MActivation( kdS )
        # Else this is an intermediate layer
        else :
            # Calculate the Mean
            kdM = kdS.sum( axis = 0 ) / float( kiN )    
            # Calculate the Variance
            kdV = ( ( kdS - kdM ) ** 2 ).sum( axis = 0 ) / float( kiN )
            # Calculate Mean Adjust Batch Sum
            kdSh = ( kdS - kdM ) / ( ( kdV + 1e-8 ) ** 0.5 )
            # Calculate Batch Normalized Output
            kdSb = ( aorSelf.vdScale * kdSh ) + aorSelf.vdOffset
            # Pass Batch Normalized Output to Activation Function
            kdA = aorSelf.MActivation( kdS ) #b )
            # Save calculations
            aorSelf.vdM  = kdM  # Store Batch Mean
            aorSelf.vdV  = kdV  # Store Batch Variance
            aorSelf.vdSh = kdSh # Store Mean Adjusted Batch Sums
            aorSelf.vdSb = kdSb # Store Batch Normalized Sums            

        # Return Output
        return( aorSelf.vdA )
    
    def MSummation( aorSelf, adX ) :
        kiN = len( adX )
        kdS = voNP.zeros( ( kiN, aorSelf.vdW.shape[ 0 ], 1 ) )
        for kiI in range( kiN ) :
            kdS[ kiI ] = voNP.dot( aorSelf.vdW, adX[ kiI ] ) + aorSelf.vdB
        # Store Sums
        aorSelf.vdS = kdS
        return( kdS )

    def MActivation( aorSelf, adX ) :
        kiN  = len( adX )
        kdA  = voNP.zeros( ( kiN, aorSelf.vdW.shape[ 0 ], 1 ) )
        kdAd = voNP.zeros( ( kiN, aorSelf.vdW.shape[ 0 ], 1 ) )
        for kiI in range( kiN ) :
            # Tan Hyperbolic
            if aorSelf.veAct == TcTypeActivation.XeTanH :
                kdA[ kiI ]  = aorSelf.MTanH( adX[ kiI ] )
                kdAd[ kiI ] = 1 - ( kdA[ kiI ] * kdA[ kiI ] )
            # Rectified Linear Unit
            elif aorSelf.veAct == TcTypeActivation.XeRELU :
                kdA[ kiI ] = aorSelf.MRELU( adX[ kiI ] )
                kdAd[ kiI ] = 1.0 * ( kdA[ kiI ] > 0 )
            # SoftMax
            elif aorSelf.veAct == TcTypeActivation.XeSoftMax :
                kdA[ kiI ] = aorSelf.MSoftMax( adX[ kiI ] )
                kdAd[ kiI ] = None
            # Sigmoid
            else : # aorSelf.veAct == TeTypeActivation.XeSigmoid
                kdA[ kiI ] = aorSelf.MSigmoid( adX[ kiI ] )
                kdAd[ kiI ] = kdA[ kiI ] * ( 1 - kdA[ kiI ] )

        # Store Calculations
        aorSelf.vdA  = kdA
        aorSelf.vdAd = kdAd

        return( kdA )

    def MSigmoid( aorSelf, adActual ) :
        return( 1 / ( 1 + voNP.exp( -adActual ) ) )

    def MTanH( aorSelf, adActual ) :        
        return( voNP.tanh( adActual ) )

    def MRELU( aorSelf, adActual ) :
        return( voNP.maximum( 0, adActual ) )

    def MSoftMax( aorSelf, adActual ) :
        kdE = voNP.exp( adActual )
        return( kdE / kdE.sum( ) )

    def MBackpropagate( aorSelf, adLR, aiBatch ) :
        kdB1  = aorSelf.vdB1        # Obtain Beta 1
        kdB2  = aorSelf.vdB2        # Obtain Beta 2
        kdWg  = aorSelf.vdWg        # Obtain Gradient for Weights
        kdWg2 = aorSelf.vdWg ** 2   # Obtain Squared Gradient for Weights
        kdBg  = aorSelf.vdBg        # Obtain Gradient for Biases
        kdBg2 = aorSelf.vdBg ** 2   # Obtain Squared Gradient for Biases
        kdE   = 1e-8                # Define small value for epsilon

        # Increment Time T
        aorSelf.vdT = aorSelf.vdT + 1

        # Update past and past squared gradients using ADAM
        aorSelf.vdWm = ( kdB1 * aorSelf.vdWm ) + ( ( 1 - kdB1 ) * kdWg )
        aorSelf.vdWv = ( kdB2 * aorSelf.vdWv ) + ( ( 1 - kdB2 ) * kdWg2 )
        aorSelf.vdBm = ( kdB1 * aorSelf.vdBm ) + ( ( 1 - kdB1 ) * kdBg )
        aorSelf.vdBv = ( kdB2 * aorSelf.vdBv ) + ( ( 1 - kdB2 ) * kdBg2 )

        # Calculate bias-corrected first and second moments
        kdWmh = aorSelf.vdWm / ( 1 - ( kdB1 ) ) # ** aorSelf.vdT ) )
        kdWvh = aorSelf.vdWv / ( 1 - ( kdB2 ) ) # ** aorSelf.vdT ) )
        kdBmh = aorSelf.vdBm / ( 1 - ( kdB1 ) ) # ** aorSelf.vdT ) )
        kdBvh = aorSelf.vdBv / ( 1 - ( kdB2 ) ) # ** aorSelf.vdT ) )      
        
        # Update Weights 
        aorSelf.vdW = aorSelf.vdW - ( adLR / ( ( kdWvh ** 0.5 ) + kdE ) * kdWmh ) # ADAM optimzation
        aorSelf.vdB = aorSelf.vdB - ( adLR / ( ( kdBvh ** 0.5 ) + kdE ) * kdBmh ) # using ADAM optimzation
        # aorSelf.vdW = aorSelf.vdW - ( adLR * aorSelf.vdWg / float( aiBatch ) )  # No Optimization    
        # aorSelf.vdB = aorSelf.vdB - ( adLR * aorSelf.vdBg / float( aiBatch ) )  # No Optimization

        # Zero the Gradients
        aorSelf.vdWg = voNP.zeros( aorSelf.vdW.shape )
        aorSelf.vdBg = voNP.zeros( ( aorSelf.vdW.shape[ 0 ], 1 ) )
