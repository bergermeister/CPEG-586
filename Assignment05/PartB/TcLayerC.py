import numpy as voNP
from TePool import TePool
from TeActivation import TeActivation

# Convolutional Neural Network Layer
class TcLayerC( object ) :
   def __init__( aorSelf, aorShape, aePool, aeActivation ) :
      # Save the shape
      aorSelf.voShape = aorShape


