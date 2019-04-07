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
      aorSelf.vdConvolSums = voNP.array( ( aorSelf.viSizeBatch, aorSelf.viNumFMThis ) )
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
            aorSelf.vdConvResults[ aiI ][ kiP ][ kiQ ] = adX[ kiP ].convolve( aorSelf.voKernels[ kiP ][ kiQ ] )
      
      // add convolution results
      for (int q = 0; q < FeatureMapList.Count; q++)
      //Parallel.For(0, FeatureMapList.Count, (q) => 
      {
            ConvolSums[batchIndex,q].Clear();
            for (int p = 0; p < PrevLayerOutputList.Count; p++)
            {
               ConvolSums[batchIndex,q] = ConvolSums[batchIndex,q] + ConvolResults[batchIndex,p, q];
            }
      }
      // evaluate each feature map i.e., perform activation after adding bias
      for(int i = 0; i < FeatureMapList.Count;i++)
      {
            FeatureMapList[i].Evaluate(ConvolSums[batchIndex,i],batchIndex);
      }
            
