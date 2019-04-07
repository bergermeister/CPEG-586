import numpy as voNP
from TeActivation import TeActivation

# Neural Network Layer
class TcLayer( object ) :
   def __init__( aorSelf, aorShape, aeActivation, abLast, adDropOut = 1.0, adMomentum = 0.8 ) :
      # Save the shape (Number of Neurons, Number of Inputs)
      aorSelf.voShape = aorShape
      aorSelf.viSizeNeurons = aorShape[ 0 ]  # Number of Neurons ( Outputs )
      aorSelf.viSizeInput   = aorShape[ 1 ]  # Number of Inputs
      aorSelf.viSizeBatch   = aorShape[ 2 ]  # Size of batch

      # Initialize weights, biases, weight gradients, and bias gradients
      aorSelf.vdW  = voNP.random.uniform( low=-0.1, high=0.1, size=( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) )
      aorSelf.vdB  = voNP.random.uniform( low=-1, high=1, size=( aorShape[ 0 ], 1 ) )
      aorSelf.vdWg = voNP.zeros( aorShape )
      aorSelf.vdBg = voNP.zeros( ( aorShape[ 0 ], 1 ) )

      # Initialize Adaptive Moment Estimation (Adam) variables
      aorSelf.vdWm = voNP.zeros( ( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) ) # Past Gradient
      aorSelf.vdWv = voNP.zeros( ( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) ) # Past Squared Gradient
      aorSelf.vdBm = voNP.zeros( ( aorShape[ 0 ], 1 ) )                           # Past Gradient
      aorSelf.vdBv = voNP.zeros( ( aorShape[ 0 ], 1 ) )                           # Past Squared Gradient
      aorSelf.vdKm = 0.0                                                          # Past Gradient
      aorSelf.vdKv = 0.0                                                          # Past Squared Gradient
      aorSelf.vdOm = 0.0                                                          # Past Gradient
      aorSelf.vdOv = 0.0                                                          # Past Squared Gradient
      aorSelf.vdB1 = 0.9
      aorSelf.vdB2 = 0.999
      aorSelf.vdT  = 0

      # Store last layer flag, drop out rate, and activation type
      aorSelf.veAct     = aeActivation
      aorSelf.vbLast    = abLast
      aorSelf.vdDropOut = adDropOut

      # Initialize Batch Normalization And Standard Neuron Layer Members
      aorSelf.vdK = 1.0                                       # Scale Factor  (Gamma)
      aorSelf.vdO = 0.0                                       # Offset Factor (Beta)
      aorSelf.vdKg = 0.0                                      # Scale Factor Gradient
      aorSelf.vdOg = 0.0                                      # Offset Factor Gradient
      aorSelf.vdM  = voNP.zeros( ( aorShape[ 0 ], 1 ) )       # Batch Mean
      aorSelf.vdMr = 1.0                                      # Batch Mean Running
      aorSelf.vdV  = voNP.zeros( ( aorShape[ 0 ], 1 ) )       # Batch Variance
      aorSelf.vdVr = 1.0                                      # Batch Variance Running
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
      if( aorSelf.vbLast ) :
         # Last Layer does not have Batch Normalized Sums
         aorSelf.vdSh = None
         aorSelf.vdSb = None
         # Pass Sums to Activation Function
         kdA = aorSelf.MActivation( kdS )
      elif( kiN == 1 ) :
         # Calculate the Mean  
         kdM = aorSelf.vdMr
         # Calculate the Variance
         kdV = aorSelf.vdVr
         # Calculate Mean Adjust Batch Sum
         kdSh = ( kdS - kdM ) / ( ( kdV + 1e-8 ) ** 0.5 )
         # Calculate Batch Normalized Output
         kdSb = ( aorSelf.vdK * kdSh ) + aorSelf.vdO
         # Pass Batch Normalized Output to Activation Function
         kdA = aorSelf.MActivation( kdSb )
         # Else this is an intermediate layer
      else :
         # Calculate the Mean  
         kdM = voNP.mean( kdS, axis = 0 ) 
         # Calculate the Variance
         kdV = voNP.var( kdS, axis = 0 )
         # Calculate Mean Adjust Batch Sum
         kdSh = ( kdS - kdM ) / ( ( kdV + 1e-8 ) ** 0.5 )
         # Calculate Batch Normalized Output
         kdSb = ( aorSelf.vdK * kdSh ) + aorSelf.vdO
         # Pass Batch Normalized Output to Activation Function
         kdA = aorSelf.MActivation( kdSb )
         # Save calculations
         aorSelf.vdM  = kdM  # Store Batch Mean
         aorSelf.vdV  = kdV  # Store Batch Variance
         aorSelf.vdSh = kdSh # Store Mean Adjusted Batch Sums
         aorSelf.vdSb = kdSb # Store Batch Normalized Sums            
         aorSelf.vdMr = ( 0.9 * aorSelf.vdMr ) + ( 0.1 * kdM.sum( axis = 0 ) )
         aorSelf.vdVr = ( 0.9 * aorSelf.vdVr ) + ( 0.1 * kdV.sum( axis = 0 ) )

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
      kdKg  = aorSelf.vdKg
      kdKg2 = aorSelf.vdKg ** 2
      kdOg  = aorSelf.vdOg
      kdOg2 = aorSelf.vdOg ** 2
      kdE   = 1e-8                # Define small value for epsilon

      # Increment Time T
      aorSelf.vdT = aorSelf.vdT + 1

      # Update past and past squared gradients using ADAM
      aorSelf.vdWm = ( kdB1 * aorSelf.vdWm ) + ( ( 1 - kdB1 ) * kdWg )
      aorSelf.vdWv = ( kdB2 * aorSelf.vdWv ) + ( ( 1 - kdB2 ) * kdWg2 )
      aorSelf.vdBm = ( kdB1 * aorSelf.vdBm ) + ( ( 1 - kdB1 ) * kdBg )
      aorSelf.vdBv = ( kdB2 * aorSelf.vdBv ) + ( ( 1 - kdB2 ) * kdBg2 )
      aorSelf.vdKm = ( kdB1 * aorSelf.vdKm ) + ( ( 1 - kdB1 ) * kdKg )
      aorSelf.vdKv = ( kdB2 * aorSelf.vdKv ) + ( ( 1 - kdB2 ) * kdKg2 )
      aorSelf.vdOm = ( kdB1 * aorSelf.vdOm ) + ( ( 1 - kdB1 ) * kdOg )
      aorSelf.vdOv = ( kdB2 * aorSelf.vdOv ) + ( ( 1 - kdB2 ) * kdOg2 )

      # Calculate bias-corrected first and second moments
      kdWmh = aorSelf.vdWm / ( 1 - ( kdB1 ** aorSelf.vdT ) )
      kdWvh = aorSelf.vdWv / ( 1 - ( kdB2 ** aorSelf.vdT ) )
      kdBmh = aorSelf.vdBm / ( 1 - ( kdB1 ** aorSelf.vdT ) )
      kdBvh = aorSelf.vdBv / ( 1 - ( kdB2 ** aorSelf.vdT ) )     
      kdKmh = aorSelf.vdKm / ( 1 - ( kdB1 ** aorSelf.vdT ) )
      kdKvh = aorSelf.vdKv / ( 1 - ( kdB2 ** aorSelf.vdT ) )      
      kdOmh = aorSelf.vdOm / ( 1 - ( kdB1 ** aorSelf.vdT ) )
      kdOvh = aorSelf.vdOv / ( 1 - ( kdB2 ** aorSelf.vdT ) )     
        
      # Update Weights 
      aorSelf.vdW = aorSelf.vdW - ( adLR / ( ( kdWvh ** 0.5 ) + kdE ) * kdWmh ) # ADAM optimzation
      aorSelf.vdB = aorSelf.vdB - ( adLR / ( ( kdBvh ** 0.5 ) + kdE ) * kdBmh ) # using ADAM optimzation
      aorSelf.vdK = aorSelf.vdK - ( adLR / ( ( kdKvh ** 0.5 ) + kdE ) * kdKmh ) # ADAM optimzation
      aorSelf.vdO = aorSelf.vdO - ( adLR / ( ( kdOvh ** 0.5 ) + kdE ) * kdOmh ) # using ADAM optimzation
      # aorSelf.vdW = aorSelf.vdW - ( adLR * aorSelf.vdWg / float( aiBatch ) )  # No Optimization    
      # aorSelf.vdB = aorSelf.vdB - ( adLR * aorSelf.vdBg / float( aiBatch ) )  # No Optimization
      # aorSelf.vdK = aorSelf.vdK - ( adLR * aorSelf.vdKg / float( aiBatch ) ) # ADAM optimzation
      # aorSelf.vdO = aorSelf.vdO - ( adLR * aorSelf.vdOg / float( aiBatch ) ) # using ADAM optimzation

      # Zero the Gradients
      aorSelf.vdWg = voNP.zeros( aorSelf.vdW.shape )
      aorSelf.vdBg = voNP.zeros( ( aorSelf.vdW.shape[ 0 ], 1 ) )

