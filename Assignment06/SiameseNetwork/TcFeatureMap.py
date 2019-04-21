import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation
from TcMatrix import TcMatrix

# Convolutional Neural Network Feature Map
class TcFeatureMap( object ) :
   def __init__( aorSelf, aiInputSize, aePool, aeActivation ) :
      aorSelf.viInputSize  = aiInputSize
      aorSelf.vePool       = aePool
      aorSelf.veActivation = aeActivation
      aorSelf.voDeltaSS    = TcMatrix( 1, 1 )
      aorSelf.voDeltaCV    = TcMatrix( 1, 1 )
      aorSelf.voOutputSS   = TcMatrix( 1, 1 )
      aorSelf.voActCV      = TcMatrix( 1, 1 )
      aorSelf.voAPrime     = TcMatrix( 1, 1 )
      aorSelf.voSum        = TcMatrix( 1, 1 )
      aorSelf.vdBias       = voNP.random.uniform( low=-0.1, high=0.1 )
      aorSelf.vdGb         = 0.0                                       # Bias Gradient

   def MForwardPass( aorSelf, adX ) :
      # Add the bias to the input data
      aorSelf.voSum = TcMatrix( adX.viRows, adX.viCols )
      aorSelf.voSum.vdData = adX.vdData + aorSelf.vdBias

      # Create matrices for the activation function and delta
      aorSelf.voActCV  = TcMatrix( aorSelf.voSum.viRows, aorSelf.voSum.viCols )
      aorSelf.voAPrime = TcMatrix( aorSelf.voSum.viRows, aorSelf.voSum.viCols )

      # Apply Activation function
      if( aorSelf.veActivation == TeActivation.XeSigmoid ) :
         aorSelf.voActCV.vdData = TeActivation.MSigmoid( aorSelf.voSum.vdData )
         aorSelf.voAPrime.vdData = 1 - ( aorSelf.voActCV.vdData ** 2 )
      elif( aorSelf.veActivation == TeActivation.XeRELU ) :
         # No APrime for RELU, delta is made zero for negative sums
         aorSelf.voActCV.vdData = TeActivation.MRELU( aorSelf.voSum.vdData )
         aorSelf.voAPrime = None

      # Apply pooling
      if( aorSelf.vePool == TePool.XeAvg ) :
         koRes = TePool.MAverage( aorSelf.voActCV )
      elif( aorSelf.vePool == TePool.XeMax ) :
         koRes = TePool.MMax( aorSelf.voActCV )
      else :
         koRes = aorSelf.voActCV

      # Record the result
      aorSelf.voOutputSS = koRes

      return( koRes )