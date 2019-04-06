import numpy as voNP
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
      aorSelf.vdConvResults = voNP.array( ( aorSelf.viSizeBatch, aorSelf.viNumFMPrev, aorSelf.viNumFMThis ) )
      aorSelf.vdConvolSums = voNP.array( ( aorSelf.viSizeBatch, aorSelf.NumFMThis ) )
      for kiB in range( aorSelf.viSizeBatch ) :
         for kiF in range( aorSelf.numFMthis ) :
            aorSelf.vdConvolSums = voNP.zeros( ( aorSelf.vdConvOutputSize, aorSelf.vdConvOutputSize ) )

      aorSelf.voKernels = voNP.array( ( aorSelf.viNumFMPrev, aorSelf.viNumFMThis ) )
      aorSelf.voKernelsG = voNP.array( ( aorSelf.viNumFMPrev, aorSelf.viNumFM ) )
      for kiP in range( aorSelf.viNumFMPrev ) :
         for kiT in range( aorSelf.viNumFMThis ) :
            aorSelf.voKernelsG[ kiP ][ kiT ] = voNP.zeros( ( aorSelf.viSizeKernl, aorSelf.viSizeKernl ) )
            aorSelf.voKernels[ kiP][ kiT ] = voNP.random.uniform( low=-0.1, 
                                                                  high=0.1, 
                                                                  size=( aorSelf.viSizeKernl, aorSelf.viSizeKernl ) )

   def MForwardPass( aorSelf, kdX, kiI ) :
      kiSizeIn  = aorSelf.voShape[ 1 ]
      kiSizeOut = aorSelf.voShape[ 0 ]
      #for kiP in range( kiSizeIn ) :
      #   for kiQ in range( kiSizeOut ) :
            
