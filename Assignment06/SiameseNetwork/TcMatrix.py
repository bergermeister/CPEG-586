import numpy as voNP
#import scipy.ndimage as voSP
import scipy.signal as voSP

class TcMatrix( object ) :
   def __init__( aorSelf, aiRows, aiCols ) :
      aorSelf.viRows = int( aiRows )
      aorSelf.viCols = int( aiCols )
      aorSelf.vdData = voNP.zeros( ( aorSelf.viRows, aorSelf.viCols ) )

   def MRandomize( aorSelf, aiMin, aiMax ) :
      aorSelf.vdData = voNP.random.uniform( low=aiMin, high=aiMax, 
                                            size=( aorSelf.viRows, aorSelf.viCols ) )

   def MClear( aorSelf ) :
      aorSelf.vdData.fill( 0 )

   def MConvolve( aorSelf, aoKernel, aoMode = 'valid' ) :
      # Perform convolution and remove edges
      koRes = voSP.convolve2d( aorSelf.vdData, aoKernel.vdData, mode=aoMode ) #, mode='constant', cval=0.0 )

      #if( koRes.shape[ 0 ] == 6 and koRes.shape[ 1 ] == 5 ) :
      #   koRes = voNP.delete( koRes, ( koRes.shape[ 0 ] - 1 ), 0 )

      # Create return matrix
      koRet = TcMatrix( koRes.shape[ 0 ], koRes.shape[ 1 ] )
      koRet.vdData = koRes

      return( koRet )

   def MConvolveFull( aorSelf, aoKernel ) :
      # Calculate the pad size
      kiPad = int( aoKernel.viRows / 2 )

      # Pad the input data
      koPad = voNP.pad( aorSelf.vdData, \
                        ( ( kiPad, kiPad ), \
                          ( kiPad, kiPad ) ), \
                        'constant', \
                        constant_values=( 0, 0 ) )

      # Perform convolution
      koRes = voSP.convolve( aorSelf.vdData, aoKernel.vdData )
      
      # Create return matrix
      koRet = TcMatrix( koRes.shape[ 0 ], koRes.shape[ 1 ] )
      koRet.vdData = koRes

      return( koRet ) 

   def MRotate90( aorSelf ) :
      koRes = TcMatrix( aorSelf.viRows, aorSelf.viCols )
      koRes.vdData = voNP.rot90( aorSelf.vdData )
      #for kiR in range( aorSelf.viRows ) :
      #   for kiC in range( aorSelf.viCols ) :
      #      koRes.vdData[ kiR ][ kiC ] = aorSelf.vdData[ aorSelf.viRows - kiC - 1 ][ kiR ]
      return( koRes )
