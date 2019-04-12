import numpy as voNP
from TeActivation import TeActivation

# Neural Network Layer
class TcLayer( object ) :
   def __init__( aorSelf, aorShape, aeActivation, adDropOut = 1.0, adMomentum = 0.8 ) :
      # Save the shape (Number of Neurons, Number of Inputs) and momentum
      aorSelf.voShape = aorShape
      aorSelf.viSizeNeurons = aorShape[ 0 ]  # Number of Neurons ( Outputs )
      aorSelf.viSizeInput   = aorShape[ 1 ]  # Number of Inputs
      aorSelf.viSizeBatch   = aorShape[ 2 ]  # Size of batch
      
      # Save the activation, dropout rate, and momentum
      aorSelf.veActivation  = aeActivation
      aorSelf.vdDropOut     = adDropOut
      aorSelf.vdMomentum    = adMomentum

      # Initialize weights, biases, weight gradients, and bias gradients
      aorSelf.vdW  = voNP.random.uniform( low=-0.1, high=0.1, size=( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) )
      aorSelf.vdB  = voNP.random.uniform( low=-1, high=1, size=( aorSelf.viSizeNeurons, 1 ) )
      aorSelf.vdWg = voNP.zeros( shape=( aorSelf.viSizeNeurons, aorSelf.viSizeInput ) )
      aorSelf.vdBg = voNP.zeros( shape=( aorSelf.viSizeNeurons, 1 ) )

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
     
      aorSelf.vdD  = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # Delta
      aorSelf.vdGw = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, aorSelf.viSizeInput ) ) # Weight Gradient
      aorSelf.vdGb = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # Bias Gradient
      aorSelf.vdS  = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # Sum
      aorSelf.vdDr = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # Drop matrix
      aorSelf.vdA  = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # A
      aorSelf.vdAp = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viSizeNeurons, 1 ) )                   # A prime


   def MForwardPass( aorSelf, adX, aiB, abBNUse, abBNTest = False ) :
      # Calculate the Sum
      aorSelf.vdS[ aiB ] = voNP.dot( aorSelf.vdW, adX ) + aorSelf.vdB

      # Batch Mean (vdMu) must be provided prior to calling this routine
      # Batch Inverse Variance (vdVi) must be provided prior to calling this routine

      # Apply Batch Normalization
      if( abBNUse == True ) :
         if( abBNTest == False ) :
            kdSm = aorSelf.vdS[ aiB ] - aorSelf.vdMu                          # Calculate Mean Adjusted Sum
            aorSelf.vdXh[ aiB ] = voNP.multiply( kdSm, aorSelf.vdVi )         # Multiply by the inverse variance
         else :
            kdSm = aorSelf.vdS[ aiB ] - aorSelf.vdRm                          # Calculate Mean Adjusted Sum
            aorSelf.vdXh[ aiB ] = kdSm / ( ( aorSelf.vdRv + 1e-8 ) ** 0.5 )   # Adjust for variance
         aorSelf.vdS[ aiB ] = voNP.multiply( aorSelf.vdXh, aorSelf.vdGamma ) + aorSelf.vdBeta

      # Apply Dropout
      if( aorSelf.vdDropOut < 1.0 ) :
         aorSelf.vdDr[ aiB ] = aorSelf.MInitializeDropout( )
         aorSelf.vdS[ aiB ] = voNP.multiply( aorSelf.vdS[ aiB ], aorSelf.vdDr[ aiB ] )

      # Apply Activation Function
      if( aorSelf.veActivation == TeActivation.XeSigmoid ) :
         aorSelf.vdA[ aiB ] = TeActivation.MSigmoid( aorSelf.vdS[ aiB ] )
         aorSelf.vdAp[ aiB ] =  aorSelf.vdA[ aiB ] * ( 1 - aorSelf.vdA[ aiB ] )
      elif( aorSelf.veActivation == TeActivation.XeRELU ) :
         aorSelf.vdA[ aiB ] = TeActivation.MRELU( aorSelf.vdS[ aiB ] )
      elif( aorSelf.veActivation == TeActivation.XeSoftMax ) :
         aorSelf.vdA[ aiB ] = TeActivation.MSoftMax( aorSelf.vdS[ aiB ] )

      return( aorSelf.vdA[ aiB ] )
    
   def MInitializeDropout( aorSelf ) :
      kdDrp = voNP.random.uniform( low=0.0, high=1.0, size=( aorSelf.viSizeNeurons, 1 ) )
      for kiR in range( kdDrp.shape[ 0 ] ) :
         for kiC in range( kdDrp.shape[ 1 ] ) :
            if( kdDrp[ kiR ][ kiC ] < aorSelf.vdDropOut ) :
               kdDrp[ kiR ][ kiC ] = 1.0 / aorSelf.vdDropOut
            else :
               kdDrp[ kiR ][ kiC ] = 0.0

      return( kdDrp )

