import numpy as voNP
from scipy import ndimage
from TePool import TePool
from TeActivation import TeActivation
from TcFeatureMap import TcFeatureMap
from TcMatrix import TcMatrix

# Convolutional Neural Network Layer
class TcLayerC( object ) :
   def __init__( aorSelf, aorShapeFMap, aorShapeSize, aePool, aeActivation ) :
      # Decode and save the shape information
      aorSelf.viNumFMThis = aorShapeFMap[ 0 ]   # Number of Feature Maps in this layer
      aorSelf.viNumFMPrev = aorShapeFMap[ 1 ]   # Number of Feature Maps in previous layer
      aorSelf.viSizeInput = aorShapeSize[ 0 ]   # Size of inputs
      aorSelf.viSizeKernl = aorShapeSize[ 1 ]   # Size of Kernel
      aorSelf.viSizeBatch = aorShapeSize[ 2 ]   # Size of Batch

      # Calculate the size of the Convolution Output: Input - Kernel + 1
      aorSelf.viConvOutputSize = aorSelf.viSizeInput - aorSelf.viSizeKernl + 1

      # Create Feature Maps
      aorSelf.voFM = [ ]
      for kiI in range( aorSelf.viNumFMThis ) :
         aorSelf.voFM.append( TcFeatureMap( aorSelf.viConvOutputSize, aePool, aeActivation, aorSelf.viSizeBatch ) )

      # Initialize Convolution Results and Sums matrices
      aorSelf.vdConvResults = voNP.ndarray( shape=( aorSelf.viSizeBatch, aorSelf.viNumFMPrev, aorSelf.viNumFMThis ), dtype=TcMatrix )
      aorSelf.vdConvolSums  = voNP.ndarray( shape=( aorSelf.viSizeBatch, aorSelf.viNumFMThis ), dtype=TcMatrix )
      for kiB in range( aorSelf.viSizeBatch ) :
         for kiF in range( aorSelf.viNumFMThis ) :
            koSum = TcMatrix( aorSelf.viConvOutputSize, aorSelf.viConvOutputSize )
            koSum.vdData = voNP.zeros( ( koSum.viRows, koSum.viCols ) )
            aorSelf.vdConvolSums[ kiB ][ kiF ] = koSum

      # Initialize Convolution Kernels
      aorSelf.voKernels  = voNP.ndarray( shape=( aorSelf.viNumFMPrev, aorSelf.viNumFMThis ), dtype=TcMatrix )
      aorSelf.voKernelsG = voNP.ndarray( shape=( aorSelf.viSizeBatch, aorSelf.viNumFMPrev, aorSelf.viNumFMThis ), dtype=TcMatrix )
      for kiP in range( aorSelf.viNumFMPrev ) :       # For each feature map in previous layer
         for kiT in range( aorSelf.viNumFMThis ) :    # For each feature map in this layer
            koKernel  = TcMatrix( aorSelf.viSizeKernl, aorSelf.viSizeKernl )
            koKernel.vdData = voNP.random.uniform( low=-0.1, high=0.1, size=( koKernel.viRows, koKernel.viCols ) )
            aorSelf.voKernels[ kiP ][ kiT ]  = koKernel
            for kiB in range( aorSelf.viSizeBatch ) :
               koKernelG = TcMatrix( aorSelf.viSizeKernl, aorSelf.viSizeKernl )
               koKernelG.vdData = voNP.zeros( ( koKernelG.viRows, koKernelG.viCols ) )
               aorSelf.voKernelsG[ kiB ][ kiP ][ kiT ] = koKernelG

   def MForwardPass( aorSelf, aoX, aiB ) :
      for kiP in range( aorSelf.viNumFMPrev ) :                # For each feature map from the previous layer
         for kiQ in range( aorSelf.viNumFMThis ) :             # For each feature map in this layer
            # Perform convolution with output of feature map from previous layer with feature in this layer
            aorSelf.vdConvResults[ aiB ][ kiP ][ kiQ ] = aoX[ kiP ].MConvolve( aorSelf.voKernels[ kiP ][ kiQ ] )            
      
      for kiQ in range( aorSelf.viNumFMThis ) :
         koSum = aorSelf.vdConvolSums[ aiB ][ kiQ ].vdData
         koSum.fill( 0 )
         for kiP in range( len( aoX ) ) :
            # Add convolution results
            koSum += aorSelf.vdConvResults[ aiB ][ kiP ][ kiQ ].vdData
         #aorSelf.vdConvolSums[ aiB ][ kiQ ] = koSum
      
      # Evaluate each feature map
      for kiI in range( aorSelf.viNumFMThis ) :
         aorSelf.voFM[ kiI ].MForwardPass( aorSelf.vdConvolSums[ aiB ][ kiI ], aiB )
      
            
