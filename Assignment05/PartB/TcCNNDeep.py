import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from TcMatrix import TcMatrix
from TeActivation import TeActivation

# Deep Convolutional Neural Network
class TcCNNDeep( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers, aiSizeBatch ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC   = aorLayersC
      aorSelf.voLayersN   = aorLayers
      aorSelf.viSizeBatch = aiSizeBatch
      aorSelf.voFlatten   = voNP.ndarray( shape=( aorSelf.viSizeBatch, 1 ), dtype=TcMatrix )

   def MForwardPass( aorSelf, adX, aiB ) :
      kiCountC = len( aorSelf.voLayersC )

      # Initialize empty list of outputs
      kdPrevOut = [ ]

      # Forward Pass on each CNN layer
      for kiI in range( kiCountC ) :
         # Build the list of outputs from the previous layer
         if( kiI == 0 ) :
            kdPrevOut.append( adX )
         else :
            kdPrevOut.clear( )
            for kiJ in range( len( aorSelf.voLayersC[ kiI - 1 ].voFM ) ) :
               kdPrevOut.append( aorSelf.voLayersC[ kiI - 1 ].voFM[ kiJ ].voOutputSS[ aiB ][ 0 ] )

         # Forward pass on the CNN Layer
         aorSelf.voLayersC[ kiI ].MForwardPass( kdPrevOut, aiB )

      # Flatten each feature map in the CNN Layer and assemble all maps into an nx1 vector
      kiSizeOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].voOutputSS[ aiB ][ 0 ].viRows   # Get the size of the feature map output
      kiSizeFlat = ( kiSizeOut ** 2 ) * len( aorSelf.voLayersC[ kiCountC - 1 ].voFM )          # Calculate the size of the flattened vector
      aorSelf.voFlatten[ aiB ][ 0 ] = TcMatrix( kiSizeFlat, 1 )                                # Create the flattened vector
      kiF = 0
      for kiI in range( len( aorSelf.voLayersC[ kiCountC - 1 ].voFM ) ) :              # For each feature map in the last layer
         koOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ kiI ].voOutputSS[ aiB ][ 0 ] # Obtain the output of the feature map
         kdFlat = koOut.vdData.flatten( )                                              # Flatten the output of the feature map
         for kiR in range( len( kdFlat ) ) :                                           # For each row in the flattened output
            aorSelf.voFlatten[ aiB ][ 0 ].vdData[ kiF ][ 0 ] = kdFlat[ kiR ]
            kiF += 1

      for kiI in range( len( aorSelf.voLayersN ) ) :
         if( kiI == 0 ) :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( aorSelf.voFlatten[ aiB ][ 0 ].vdData, aiB, False )
         else :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( kdRes, aiB, False )

      # Return result
      return( kdRes )

   def MTrain( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
      for kiEpoch in range( aiEpochs ) :
         # Shuffle the input/output pairs
         kdX, kdY = voShuffle( adX, adY, random_state = 0 )

         for kiI in range( 0, len( kdX ), aiBatchSize ) :
            with concurrent.futures.ProcessPoolExecutor() as executor :
               for kiB in range( aiBatchSize ) :
                  kiJ = kiI + kiB
                  koX = TcMatrix( kdX[ kiJ ].shape[ 0 ], kdX[ kiJ ].shape[ 1 ] )
                  koX.vdData = kdX[ kiJ ]
                  kdA = aorSelf.MForwardPass( koX, kiB )

                  # Calculate the loss
                  kdL = ( kdA - kdY[ kiJ ] ) ** 2

                  aorSelf.MBackPropagate( kdY[ kiJ ], kiB )

   def MBackPropagate( aorSelf, aiY, aiB ) :
      kiCountLn = len( aorSelf.voLayersN )   # Number of NN Layers
      kiCountLc = len( aorSelf.voLayersC )   # Number of CNN Layers

      # Compute Deltas on regular NN layers
      for kiI in range( kiCountLn - 1, -1, -1 ) :  # For each NN Layer, starting with last and working towards first
         koLayer = aorSelf.voLayersN[ kiI ]        # Obtian a reference to the Layer
         if( kiI == ( kiCountLn - 1 ) ) :          # If Last Layer 
            koLayer.vdD[ aiB ] = -( aiY - koLayer.vdA[ aiB ] )       # Assume SoftMax
            if( koLayer.veActivation == TeActivation.XeSigmoid ) :   # If Activation is Sigmoid
               koLayer.vdD[ aiB ] = voNP.multiply( koLayer.vdD[ aiB ], koLayer.vdAp[ aiB ] )
            elif( koLayer.veActivation == TeActivation.XeRELU ) :
               koLayer.vdD[ aiB ] = 1.0 * ( koLayer.vdS[ aiB ] > 0 )
            
         else : # Previous Layer
            koLayer.vdD[ aiB ] = voNP.dot( aorSelf.voLayersN[ kiI + 1 ].vdW.T, aorSelf.voLayersN[ kiI + 1 ].vdD[ aiB ] )
            # Apply Dropout
            if( koLayer.vdDropOut < 1.0 ) :
               koLayer.vdD[ aiB ] = voNP.multiply( koLayer.vdD[ aiB ], koLayer.vdDr[ aiB ] )
         
            # Calculate Delta
            if( koLayer.veActivation == TeActivation.XeSigmoid ) :
               koLayer.vdD[ aiB ] = voNP.multiply( koLayer.vdD[ aiB ], koLayer.vdAp[ aiB ] )
            elif( koLayer.veActivation == TeActivation.XeRELU ) :
               koLayer.vdD[ aiB ] = 1.0 * ( koLayer.vdS[ aiB ] > 0 )
         
         koLayer.vdGb[ aiB ] = koLayer.vdGb[ aiB ] + koLayer.vdD[ aiB ]
            
         if( kiI == 0 ) : # First NN Layer connected to CNN last layer via flatten
            koLayer.vdGw[ aiB ] = koLayer.vdGw[ aiB ] + voNP.dot( koLayer.vdD[ aiB ], aorSelf.voFlatten[ aiB ][ 0 ].vdData.T  )
         else :
            koLayer.vdGw[ aiB ] = koLayer.vdGw[ aiB ] + voNP.dot( koLayer.vdD[ aiB ], aorSelf.voLayersN[ kiI - 1].vdA[ aiB ].T ) 
     
     # // compute delta on the output of SS (flat) layer of all feature maps
     # Matrix deltaSSFlat = this.LayerList[0].W.Transpose() * this.LayerList[0].Delta[b];
     #
     # // do reverse flattening and distribute the deltas on
     # // each feature map's SS (SubSampling layer)
     # int index = 0;
     # // last CNN layer
     # foreach (FeatureMap fmp in CNNLayerList[CNNLayerList.Count - 1].FeatureMapList)
     # {
     #       fmp.DeltaSS[b] = new Matrix(fmp.OutPutSS[b].Rows, fmp.OutPutSS[b].Cols);
     #       for (int m = 0; m < fmp.OutPutSS[b].Rows; m++)
     #       {
     #          for (int n = 0; n < fmp.OutPutSS[b].Cols; n++)
     #          {
     #             fmp.DeltaSS[b].D[m][n] = deltaSSFlat.D[index][0];
     #             index++;
     #          }
     #       }
     # }
     # // process CNN layers in reverse order, from last layer towards input
     # for (int cnnCount = CNNLayerList.Count - 1; cnnCount >= 0; cnnCount--)
     # {
     #       // compute deltas on the C layers - distrbute deltas from SS layer
     #       // then multiply by the activation function
     #       //foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
     #       //Parallel.For(0, CNNLayerList[cnnCount].FeatureMapList.Count, (k) =>
     #       for (int k = 0; k < CNNLayerList[cnnCount].FeatureMapList.Count; k++)
     #       {
     #          FeatureMap fmp = CNNLayerList[cnnCount].FeatureMapList[k];
     #          int indexm = 0; int indexn = 0;
     #          fmp.DeltaCV[b] = new Matrix(fmp.OutPutSS[b].Rows * 2, fmp.OutPutSS[b].Cols * 2);
     #          for (int m = 0; m < fmp.DeltaSS[b].Rows; m++)
     #          {
     #             indexn = 0;
     #             for (int n = 0; n < fmp.DeltaSS[b].Cols; n++)
     #             {
     #                   if (fmp.activationType == ActivationType.SIGMOID)
     #                   {
     #                      fmp.DeltaCV[b].D[indexm][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm][indexn];
     #                      fmp.DeltaCV[b].D[indexm][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm][indexn + 1];
     #                      fmp.DeltaCV[b].D[indexm + 1][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm + 1][indexn];
     #                      fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm + 1][indexn + 1];
     #                      indexn = indexn + 2;
     #                   }
     #                   if (fmp.activationType == ActivationType.RELU)
     #                   {
     #                      if (fmp.Sum[b].D[indexm][indexn] > 0)
     #                         fmp.DeltaCV[b].D[indexm][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
     #                      else
     #                         fmp.DeltaCV[b].D[indexm][indexn] = 0;
     #                      if (fmp.Sum[b].D[indexm][indexn + 1] > 0)
     #                         fmp.DeltaCV[b].D[indexm][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
     #                      else
     #                         fmp.DeltaCV[b].D[indexm][indexn + 1] = 0;
     #                      if (fmp.DeltaCV[b].D[indexm + 1][indexn] > 0)
     #                         fmp.DeltaCV[b].D[indexm + 1][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
     #                      else
     #                         fmp.DeltaCV[b].D[indexm + 1][indexn] = 0;
     #                      if (fmp.DeltaCV[b].D[indexm + 1][indexn + 1] > 0)
     #                         fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
     #                      else
     #                         fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = 0;
     #                      indexn = indexn + 2;
     #                   }
     #             }
     #             indexm = indexm + 2;
     #          }
     #       }
     #
     #       //----------compute BiasGrad in current CNN Layer-------
     #       foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
     #       {
     #          for (int u = 0; u < fmp.DeltaCV[b].Rows; u++)
     #          {
     #             for (int v = 0; v < fmp.DeltaCV[b].Cols; v++)
     #                   lock (olock)
     #                   {
     #                      fmp.BiasGrad += fmp.DeltaCV[b].D[u][v];
     #                   }
     #          }
     #       }
     #       //----------compute gradients for pxq kernels in current CNN layer--------
     #       if (cnnCount > 0)  // not the first CNN layer
     #       {
     #          for (int p = 0; p < CNNLayerList[cnnCount - 1].FeatureMapList.Count; p++)
     #          //Parallel.For(0, CNNLayerList[cnnCount - 1].FeatureMapList.Count, (p) =>
     #          {
     #             for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
     #             {
     #                   lock (olock)
     #                   {
     #                      CNNLayerList[cnnCount].KernelGrads[p, q] = CNNLayerList[cnnCount].KernelGrads[p, q] +
     #                         CNNLayerList[cnnCount - 1].FeatureMapList[p].OutPutSS[b].RotateBy90().RotateBy90().Convolution(CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b]);
     #                   }
     #             }
     #          }
     #          //---------------this layer is done, now backpropagate to prev CNN Layer----------
     #          for (int p = 0; p < CNNLayerList[cnnCount - 1].FeatureMapList.Count; p++)
     #          //Parallel.For(0, CNNLayerList[cnnCount - 1].FeatureMapList.Count, (p) =>
     #          {
     #             int size = CNNLayerList[cnnCount - 1].FeatureMapList[p].OutPutSS[b].Rows;
     #             CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = new Matrix(size, size);
     #             //CNNLayerList[cnnCount - 1].FeatureMap2List[p].DeltaSS.Clear();
     #             for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
     #             {
     #                   CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] +
     #                   CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b].ConvolutionFull(
     #                   CNNLayerList[cnnCount].Kernels[p, q].RotateBy90().RotateBy90());
     #             }
     #          }
     #       }
     #       else  // very first CNN layer which is connected to input
     #       {     // has 1xnumFeaturemaps 2-D array of Kernels and Kernel Gradients
     #          //----------compute gradient for first layer cnn kernels--------
     #          for (int p = 0; p < 1; p++)
     #          {
     #             for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
     #             {
     #                   lock (olock)
     #                   {
     #                      CNNLayerList[cnnCount].KernelGrads[p, q] = CNNLayerList[cnnCount].KernelGrads[p, q] +
     #                         InputDataList[dj + b].RotateBy90().RotateBy90().Convolution(CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b]);
     #                   }
     #             }
     #          }
     #       }
     # }