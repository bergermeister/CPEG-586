import numpy as voNP
from TcMatrix import TcMatrix

class TePool( object ) :
   XeNone = 0
   XeMax  = 1
   XeAvg  = 2

   def MAverage( adX ) :
      koRes = TcMatrix( adX.viRows / 2, adX.viCols / 2 )
      for kiR in range( koRes.viRows ) :
         for kiC in range( koRes.viCols ) :
            koRes.vdData[ kiR ][ kiC ] = ( adX.vdData[ ( kiR * 2 ) + 0 ][ ( kiC * 2 ) + 0 ] + 
                                           adX.vdData[ ( kiR * 2 ) + 0 ][ ( kiC * 2 ) + 1 ] + 
                                           adX.vdData[ ( kiR * 2 ) + 1 ][ ( kiC * 2 ) + 0 ] + 
                                           adX.vdData[ ( kiR * 2 ) + 1 ][ ( kiC * 2 ) + 1 ] ) / 4.0;
      return( koRes )

   def MMax( adX ) :
      koRes = TcMatrix( adX.viRows / 2, adX.viCols / 2 )
      kdMax = voNP.max( adX.vdData )
      for kiR in range( koRes.viRows ) :
         for kiC in range( koRes.kiCols ) :
            koRes.vdData[ kiR ][ kiC ] = kdMax
      return( koRes )