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
      koRes = TcMatrix( aorSelf.viRows, aorSelf.viCols )
      #koRes.vdData = voNP.rot90( aorSelf.vdData )
      for kiR in range( aorSelf.viRows ) :
         for kiC in range( aorSelf.viCols ) :
            koRes.vdData[ kiR ][ kiC ] = aorSelf.vdData[ aorSelf.viRows - kiC - 1 ][ kiR ]
      return( koRes )

   def MCorrelation( aorSelf, aoKernel ) :
      # No padding, assumes kernel is a square matrix, no flip of kernel
      kiK = aoKernel.viCols;
      kiM = aorSelf.viRows;
      kiN = aorSelf.viCols;

      koRes = TcMatrix( kiM - ( kiK - 1 ), kiN - ( kiK - 1 ) ); 
      for kiI in range( koRes.viRows ) :
         for kiJ in range( koRes.viCols ) :
            kdSum = 0.0

            # Iterate over kernel
            for kiKi in range( kiK ) :
               for kiKj in range( kiK ) :
                  kdData = aorSelf.vdData[ kiI + kiKi ][ kiJ + kiKj ]
                  kdVal  = aoKernel.vdData[ kiKi ][ kiKj ]
                  kdSum += kdData * kdVal

            koRes.vdData[ kiI ][ kiJ ] = kdSum;

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
