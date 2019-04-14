import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation
from TcMatrix import TcMatrix

# Convolutional Neural Network Feature Map
class TcFeatureMap( object ) :
   def __init__( aorSelf, aiInputSize, aePool, aeActivation, aiBatchSize ) :
      aorSelf.viInputSize  = aiInputSize
      aorSelf.viBatchSize  = aiBatchSize
      aorSelf.vePool       = aePool
      aorSelf.veActivation = aeActivation
      aorSelf.voDeltaSS    = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.voDeltaCV    = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.voOutputSS   = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.voActCV      = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.voAPrime     = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.voSum        = voNP.ndarray( ( aiBatchSize ), dtype=TcMatrix )
      aorSelf.vdBias       = voNP.random.uniform( low=-0.1, high=0.1 )
      aorSelf.vdGb         = 0.0    # Bias Gradient
      aorSelf.voGk         = TcMatrix( 0, 0 )      # Kernel Gradients

   def MForwardPass( aorSelf, adX, aiB ) :
      # Copy the input to the sum
      koSum = TcMatrix( adX.viRows, adX.viCols )
      koSum.vdData = adX.vdData.copy( )
      
      # Add the bias to the sum
      for kiR in range( koSum.vdData.shape[ 0 ] ) :
         for kiC in range( koSum.vdData.shape[ 1 ] ) :
            koSum.vdData[ kiR ][ kiC ] += aorSelf.vdBias
      
      # Save the sum at the batch index
      aorSelf.voSum[ aiB ] = koSum

      # Create matrices for the activation function and delta
      aorSelf.voActCV[ aiB ] = TcMatrix( koSum.viRows, koSum.viCols )
      aorSelf.voAPrime[ aiB ] = TcMatrix( koSum.viRows, koSum.viCols )

      # Apply Activation function
      if( aorSelf.veActivation == TeActivation.XeSigmoid ) :
         aorSelf.voActCV[ aiB ].vdData = TeActivation.MSigmoid( aorSelf.voSum[ aiB ].vdData )
         aorSelf.voAPrime[ aiB ].vdData = 1 - ( aorSelf.voActCV[ aiB ].vdData ** 2 )
      elif( aorSelf.veActivation == TeActivation.XeRELU ) :
         # No APrime for RELU, delta is made zero for negative sums
         aorSelf.voActCV[ aiB ].vdData = TeActivation.MRELU( aorSelf.voSum[ aiB ].vdData )
         aorSelf.voAPrime[ aiB ] = None

      # Apply pooling
      if( aorSelf.vePool == TePool.XeAvg ) :
         koRes = TePool.MAverage( aorSelf.voActCV[ aiB ] )
      elif( aorSelf.vePool == TePool.XeMax ) :
         koRes = TePool.MMax( aorSelf.voActCV[ aiB ] )
      else :
         koRes = aorSelf.voActCV[ aiB ]

      # Record the result
      aorSelf.voOutputSS[ aiB ] = koRes

      return( koRes )