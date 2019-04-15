import numpy as voNP
from TeActivation import TeActivation

# Neural Network Layer
class TcLayer( object ) :
   def __init__( aorSelf, aorShape, aeActivation, adDropOut = 1.0 ) :
      # Save the shape (Number of Neurons, Number of Inputs)
      aorSelf.voShape = aorShape
      aorSelf.viSizeNeurons = aorShape[ 0 ]  # Number of Neurons ( Outputs )
      aorSelf.viSizeInput   = aorShape[ 1 ]  # Number of Inputs
      
      # Save the activation and dropout rate
      aorSelf.veActivation  = aeActivation
      aorSelf.vdDropOut     = adDropOut

      # Initialize weights, biases, weight gradients, and bias gradients
      aorSelf.vdW  = voNP.random.uniform( low=-0.1, high=0.1, size=( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) )
      aorSelf.vdB  = voNP.random.uniform( low=-1, high=1, size=( aorSelf.viSizeNeurons, 1 ) )

      aorSelf.vdVi   = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Inverse of variance
      aorSelf.vdMu   = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Batch Mean
      aorSelf.vdBeta = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) 
      aorSelf.vdGamma = voNP.ones( shape=( aorSelf.viSizeNeurons, 1 ) ) 

      aorSelf.vdXh = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) 
      aorSelf.vdDb = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Delta Beta
      aorSelf.vdDg = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Delta Gamma
      aorSelf.vdV  = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Batch Variance
      aorSelf.vdRm = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Running batch mean
      aorSelf.vdRv = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) ) # Running batch variance
     
      aorSelf.vdD  = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # Delta
      aorSelf.vdGw = voNP.zeros( shape=( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) ) # Weight Gradient
      aorSelf.vdGb = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # Bias Gradient
      aorSelf.vdS  = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # Sum
      aorSelf.vdDr = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # Drop matrix
      aorSelf.vdA  = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # A
      aorSelf.vdAp = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )                   # A prime


   def MForwardPass( aorSelf, adX, abBNUse, abBNTest = False ) :
      # Calculate the Sum
      aorSelf.vdS = voNP.dot( aorSelf.vdW, adX ) + aorSelf.vdB

      # Batch Mean (vdMu) must be provided prior to calling this routine
      # Batch Inverse Variance (vdVi) must be provided prior to calling this routine

      # Apply Batch Normalization
      if( abBNUse == True ) :
         if( abBNTest == False ) :
            kdSm = aorSelf.vdS - aorSelf.vdMu                          # Calculate Mean Adjusted Sum
            aorSelf.vdXh = voNP.multiply( kdSm, aorSelf.vdVi )         # Multiply by the inverse variance
         else :
            kdSm = aorSelf.vdS - aorSelf.vdRm                          # Calculate Mean Adjusted Sum
            aorSelf.vdXh = kdSm / ( ( aorSelf.vdRv + 1e-8 ) ** 0.5 )   # Adjust for variance
         aorSelf.vdS = voNP.multiply( aorSelf.vdXh, aorSelf.vdGamma ) + aorSelf.vdBeta

      # Apply Dropout
      if( aorSelf.vdDropOut < 1.0 ) :
         aorSelf.vdDr = aorSelf.MInitializeDropout( )
         aorSelf.vdS = voNP.multiply( aorSelf.vdS, aorSelf.vdDr )

      # Apply Activation Function
      if( aorSelf.veActivation == TeActivation.XeSigmoid ) :
         aorSelf.vdA = TeActivation.MSigmoid( aorSelf.vdS )
         aorSelf.vdAp =  aorSelf.vdA * ( 1 - aorSelf.vdA )
      elif( aorSelf.veActivation == TeActivation.XeRELU ) :
         aorSelf.vdA = TeActivation.MRELU( aorSelf.vdS )
      elif( aorSelf.veActivation == TeActivation.XeSoftMax ) :
         aorSelf.vdA = TeActivation.MSoftMax( aorSelf.vdS )

      return( aorSelf.vdA )
    
   def MInitializeDropout( aorSelf ) :
      kdDrp = voNP.random.uniform( low=0.0, high=1.0, size=( aorSelf.viSizeNeurons, 1 ) )
      for kiR in range( kdDrp.shape[ 0 ] ) :
         for kiC in range( kdDrp.shape[ 1 ] ) :
            if( kdDrp[ kiR ][ kiC ] < aorSelf.vdDropOut ) :
               kdDrp[ kiR ][ kiC ] = 1.0 / aorSelf.vdDropOut
            else :
               kdDrp[ kiR ][ kiC ] = 0.0

      return( kdDrp )

