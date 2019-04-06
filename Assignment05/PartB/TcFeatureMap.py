import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation

# Convolutional Neural Network Feature Map
class TcFeatureMap( object ) :
   def __init__( aorSelf, aiInputSize, aePool, aeActivation ) :
      aorSelf.viInputSize = aiInputSize
