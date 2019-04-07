import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation

# Convolutional Neural Network Feature Map
class TcFeatureMap( object ) :
   def __init__( aorSelf, aiInputSize, aePool, aeActivation, aiBatchSize ) :
      aorSelf.viInputSize = aiInputSize
      aorSelf.viBatchSize = aiBatchSize
      aorSelf.vePool      = aePool
      aorSelf.veAct       = aeActivation
      aorSelf.vdDeltaSS = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdDeltaCV = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdOutputSS = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdActCV = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdAPrime = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdSum = voNP.array( ( aiBatchSize, 1 ) )
      aorSelf.vdBias = voNP.random.uniform( low=-0.1, high=0.1 )


   def MForwardPass( aorSelf, adX, aiI ) :
      # Add the bias to the input
      aorSelf.vdSum[ aiI ] = adX + aorSelf.vdBias

      # Apply Activation function
      if( aorSelf.veAct == TeActivation.XeSigmoid ) :
         aorSelf.vdActCV[ aiI ] = TeActivation.MSigmoid( aorSelf.vdSum[ aiI ] )
         aorSelf.vdAPrime[ aiI ] = 1 - ( vdActCV[ aiI ] * vdActCV[ aiI ] )
      elif( aorSelf.veAct == TeActivation.XeRELU ) :
         # No APrime for RELU, delta is made zero for negative sums
         aorSelf.vdActCV[ aiI ] = TeActivation.MRELU( aorSelf.vdSum[ aiI ] )

      # Apply pooling
      if( aorSelf.vePool == TePool.XeAvg ) :
         kdRes = TePool.MAverage( aorSelf.vdActCV[ aiI ] )
      elif( aorSelf.vePool == TePool.XeMax ) :
         kdRes = TePool.MMax( aorSelf.vdActCV[ aiI ] )
      else :
         kdRes = aorSelf.vdActCV[ aiI ]

      # Record the result
      aorSelf.vdOutputSS[ aiI ] = kdRes

      return( kdRes )