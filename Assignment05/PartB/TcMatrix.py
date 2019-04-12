import numpy as voNP

class TcMatrix( object ) :
   def __init__( aorSelf, aiRows, aiCols ) :
      aorSelf.viRows = int( aiRows )
      aorSelf.viCols = int( aiCols )
      aorSelf.vdData = voNP.zeros( ( aorSelf.viRows, aorSelf.viCols ) )

   def MRandomize( aorSelf, aiMin, aiMax ) :
      aorSelf.vdData = voNP.random.uniform( low=aiMin, high=aiMax, 
                                            size=( aorSelf.viRows, aorSelf.viCols ) )

   def MConvolve( aorSelf, aoKernel ) :
      koRot = aoKernel.MRotate90( ).MRotate90( )
      return( aorSelf.MCorrelation( koRot ) )

   def MRotate90( aorSelf ) :
      koRes = TcMatrix( aorSelf.viCols, aorSelf.viRows )
      koRes.vdData = voNP.rot90( aorSelf.vdData )
      return( koRes )

   def MCorrelation( aorSelf, aoKernel ) :
      # No padding, assumes kernel is a square matrix, no flip of kernel
      kiCk = aoKernel.viCols;
      kiRm = aorSelf.viRows;
      kiCm = aorSelf.viCols;

      koRes = TcMatrix( kiRm - ( kiCk - 1 ), kiCm - ( kiCk - 1 ) ); 
      for kiR in range( koRes.viRows ) :
         for kiC in range( koRes.viCols ) :
            kdSum = 0.0

            # Iterate over kernel
            for kiI in range( kiCk ) :
               for kiJ in range( kiCk ) :
                  kdData = aorSelf.vdData[ kiR + kiI ][ kiC + kiJ ]
                  kdVal  = aoKernel.vdData[ kiI ][ kiJ ]
                  kdSum += kdData * kdVal

            koRes.vdData[ kiR ][ kiC ] = kdSum;

      return( koRes )
