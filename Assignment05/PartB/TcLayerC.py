import numpy as voNP
from scipy import ndimage
from TePool import TePool
from TeActivation import TeActivation
from TcFeatureMap import TcFeatureMap

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
      aorSelf.vdConvResults = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viNumFMPrev, aorSelf.viNumFMThis, 28, 28 ) )
      aorSelf.vdConvolSums = voNP.zeros( shape=( aorSelf.viSizeBatch, aorSelf.viNumFMThis ) )
      for kiB in range( aorSelf.viSizeBatch ) :
         for kiF in range( aorSelf.viNumFMThis ) :
            aorSelf.vdConvolSums = voNP.zeros( ( aorSelf.viConvOutputSize, aorSelf.viConvOutputSize ) )

      
      aorSelf.voKernelsG = voNP.zeros( ( aorSelf.viNumFMPrev, aorSelf.viNumFMThis, aorSelf.viSizeKernl, aorSelf.viSizeKernl ) )
      aorSelf.voKernels = voNP.random.uniform( low=-0.1, high=0.1,
                                              size=( aorSelf.viNumFMPrev, aorSelf.viNumFMThis, 
                                                     aorSelf.viSizeKernl, aorSelf.viSizeKernl ) )

   def MForwardPass( aorSelf, adX, aiI ) :
      for kiP in range( aorSelf.viNumFMPrev ) :                # For each feature map from the previous layer
         for kiQ in range( aorSelf.viNumFMThis ) :             # For each feature map in this layer
            # Perform convolution with output of feature map from previous layer with feature in this layer
            koCV = ndimage.convolve( adX[ kiP ], aorSelf.voKernels[ kiP ][ kiQ ].rot90() )
            kdCV = voNP.convolve( aorSelf.voKernels[ kiP ][ kiQ ], adX[ kiP ] )
            aorSelf.vdConvResults[ aiI ][ kiP ][ kiQ ] = voNP.convolve( adX[ kiP ], aorSelf.voKernels[ kiP ][ kiQ ] )
      
      for kiQ in range( aorSelf.viNumFMThis ) :
         aorSelf.vdConvolSums[ aiI ][ kiQ ] = 0.0
         for kiP in range( aorSelf.viNumFMPrev ) :
            # Add convolution results
            aorSelf.vdConvolSums[ aiI ][ kiQ ] += aorSelf.vdConvolResults[ aiI ][ kiP ][ kiQ ]
      
      # Evaluate each feature map
      for kiI in range( aorSelf.viNumFMThis ) :
         aorSelf.voFM[ kiI ].MForwardPass( aorSelf.vdConvolSums[ aiI ][ kiI ], aiI )
      
            
