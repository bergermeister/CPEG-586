import numpy as voNP

class TcMatrix( object ) :
   def __init__( aorSelf, aiRows, aiCols ) :
      aorSelf.viRows = int( aiRows )
      aorSelf.viCols = int( aiCols )
      aorSelf.vdData = voNP.zeros( ( aorSelf.viRows, aorSelf.viCols ) )

   def MRandomize( aorSelf, aiMin, aiMax ) :
      aorSelf.vdData = voNP.random.uniform( low=aiMin, high=aiMax, 
                                            size=( aorSelf.viRows, aorSelf.viCols ) )

   def MClear( aorSelf ) :
      aorSelf.vdData = voNP.zeros( shape=( aorSelf.viRows, aorSelf.viCols ) )

   def MConvolve( aorSelf, aoKernel ) :
      koRot = aoKernel.MRotate90( ).MRotate90( )
      return( aorSelf.MCorrelation( koRot ) )

   def MConvolveFull( aorSelf, aoKernel ) :
      koRot = aoKernel.MRotate90( ).MRotate90( )
      return( aorSelf.MCorrelationFull( koRot ) )

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

   def MCorrelationFull( aorSelf, aoKernel ) :
      # Assumes kernel is a square matrix, no flip of kernel
      kiK = aoKernel.viCols
      kiM = aorSelf.viRows
      kiN = aorSelf.viCols

      koRes = TcMatrix( kiM + ( kiK - 1 ), kiN + ( kiK - 1 ) )
      for kiI in range( kiM + ( kiK - 1 ) ) :
         for kiJ in range( kiN + ( kiK - 1 ) ) :
            kdSum = 0
            for kiKi in range( -( kiK - 1 ), 1, 1 ) : # Iterate over kernel
               for kiKj in range( -( kiK - 1 ), 1, 1 ) :
                  if( ( ( kiI + kiKi ) >= 0 ) and ( ( kiI + kiKi ) < kiM ) and ( ( kiJ + kiKj ) >= 0 ) and ( ( kiJ + kiKj ) < kiN ) ) :
                     kdData = aorSelf.vdData[ kiI + kiKi ][ kiJ + kiKj ]
                     kdKVal = aoKernel.vdData[ kiKi + ( kiK - 1 ) ][ kiKj + ( kiK - 1 ) ]
                     kdSum += kdData * kdKVal
            koRes.vdData[ kiI ][ kiJ ]= kdSum

      return( koRes )
