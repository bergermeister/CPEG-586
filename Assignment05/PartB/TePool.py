import numpy as voNP

class TePool( object ) :
   XeNone = 0
   XeMax  = 1
   XeAvg  = 2

   def MAverage( adX ) :
      kdRes = voNP.zeros( ( adX.shape[ 0 ] / 2, adX.shape[ 1 ] / 2 ) )
      for kiR in range( adX.shape[ 0 ] / 2 ) :
         for kiC in range( adX.shape[ 1 ] / 2 ) :
            kdRes[ kiR ][ kiC ] = ( adX[ ( kiR * 2 ) + 0 ][ ( kiC * 2 ) + 0 ] + 
                                    adX[ ( kiR * 2 ) + 0 ][ ( kiC * 2 ) + 1 ] + 
                                    adX[ ( kiR * 2 ) + 1 ][ ( kiC * 2 ) + 0 ] + 
                                    adX[ ( kiR * 2 ) + 1 ][ ( kIC * 2 ) + 1 ] ) / 4.0;
      return( kdRes )

   def MMax( adX ) :
      kdRes = voNP.zeros( ( adX.shape[ 0 ] / 2, adX.shape[ 1 ] / 2 ) )
      kdMax = voNP.max( adX )
      for kiR in range( adX.shape[ 0 ] / 2 ) :
         for kiC in range( adX.shape[ 1 ] / 2 ) :
            kdRes[ kiR ][ kiC ] = kdMax
      return( kdRes )