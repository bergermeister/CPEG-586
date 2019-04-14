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
      aorSelf.vdGb         = voNP.zeros( ( aiBatchSize ) )   # Bias Gradient
      #aorSelf.voGk         = None                            # Kernel Gradients

   def MForwardPass( aorSelf, adX, aiB ) :
      # Add the bias to the input data
      aorSelf.voSum[ aiB ] = TcMatrix( adX.viRows, adX.viCols )
      aorSelf.voSum[ aiB ].vdData = adX.vdData + aorSelf.vdBias

      # Create matrices for the activation function and delta
      aorSelf.voActCV [ aiB ] = TcMatrix( aorSelf.voSum[ aiB ].viRows, aorSelf.voSum[ aiB ].viCols )
      aorSelf.voAPrime[ aiB ] = TcMatrix( aorSelf.voSum[ aiB ].viRows, aorSelf.voSum[ aiB ].viCols )

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