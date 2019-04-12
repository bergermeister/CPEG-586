import numpy as voNP

class TeActivation( object ) :
   XeSigmoid = 1
   XeTanH    = 2
   XeRELU    = 3
   XeSoftMax = 4

   def MSigmoid( adActual ) :
      return( 1 / ( 1 + voNP.exp( -adActual ) ) )

   def MTanH( adActual ) :        
      return( voNP.tanh( adActual ) )

   def MRELU( adActual ) :
      return( voNP.maximum( 0, adActual ) )

   def MSoftMax( adActual ) :
      kdE = voNP.exp( adActual )
      return( kdE / kdE.sum( ) )