import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation
from TcMatrix import TcMatrix

# Convolutional Neural Network Feature Map
class TcFeatureMap( object ) :
   def __init__( aorSelf, aiInputSize, aePool, aeActivation, aiBatchSize ) :
      aorSelf.viInputSize = aiInputSize
      aorSelf.viBatchSize = aiBatchSize
      aorSelf.vePool      = aePool
      aorSelf.veAct       = aeActivation
      aorSelf.voDeltaSS  = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.voDeltaCV  = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.voOutputSS = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.voActCV    = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.voAPrime   = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.voSum      = voNP.ndarray( ( aiBatchSize, 1 ), dtype=TcMatrix )
      aorSelf.vdBias     = voNP.random.uniform( low=-0.1, high=0.1 )

   def MForwardPass( aorSelf, adX, aiB ) :
      # Copy the input to the sum
      koSum = TcMatrix( adX.viRows, adX.viCols )
      koSum.vdData = adX.vdData.copy( )
      
      # Add the bias to the sum
      for kiR in range( koSum.vdData.shape[ 0 ] ) :
         for kiC in range( koSum.vdData.shape[ 1 ] ) :
            koSum.vdData[ kiR ][ kiC ] += aorSelf.vdBias
      
      # Save the sum at the batch index
      aorSelf.voSum[ aiB ][ 0 ] = koSum

      # Create matrices for the activation function and delta
      aorSelf.voActCV[ aiB ][ 0 ] = TcMatrix( koSum.viRows, koSum.viCols )
      aorSelf.voAPrime[ aiB ][ 0 ] = TcMatrix( koSum.viRows, koSum.viCols )

      # Apply Activation function
      if( aorSelf.veAct == TeActivation.XeSigmoid ) :
         aorSelf.voActCV[ aiB ][ 0 ].vdData = TeActivation.MSigmoid( aorSelf.voSum[ aiB ][ 0 ].vdData )
         aorSelf.voAPrime[ aiB ][ 0 ].vdData = 1 - ( vdActCV[ aiB ][ 0 ].vdData ** 2 )
      elif( aorSelf.veAct == TeActivation.XeRELU ) :
         # No APrime for RELU, delta is made zero for negative sums
         aorSelf.voActCV[ aiB ][ 0 ].vdData = TeActivation.MRELU( aorSelf.voSum[ aiB ][ 0 ].vdData )
         aorSelf.voAPrime[ aiB ][ 0 ] = None

      # Apply pooling
      if( aorSelf.vePool == TePool.XeAvg ) :
         koRes = TePool.MAverage( aorSelf.voActCV[ aiB ][ 0 ] )
      elif( aorSelf.vePool == TePool.XeMax ) :
         koRes = TePool.MMax( aorSelf.voActCV[ aiB ][ 0 ] )
      else :
         koRes = aorSelf.voActCV[ aiB ][ 0 ]

      # Record the result
      aorSelf.voOutputSS[ aiB ] = koRes

      return( koRes )