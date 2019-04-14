import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from multiprocessing import Pool, freeze_support, cpu_count
from TcMatrix import TcMatrix
from TeActivation import TeActivation

# Deep Convolutional Neural Network
class TcCNNDeep( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers, aiSizeBatch ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC   = aorLayersC
      aorSelf.voLayersN   = aorLayers
      aorSelf.viSizeBatch = aiSizeBatch
      aorSelf.voFlatten   = voNP.ndarray( shape=( aorSelf.viSizeBatch ), dtype=TcMatrix )
      aorSelf.vdLoss      = voNP.zeros( shape=( aorSelf.viSizeBatch ), dtype=float )

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
            for koFM in aorSelf.voLayersC[ kiI -1 ].voFM :
               kdPrevOut.append( koFM.voOutputSS[ aiB ] )

         # Forward pass on the CNN Layer
         aorSelf.voLayersC[ kiI ].MForwardPass( kdPrevOut, aiB )

      # Flatten each feature map in the CNN Layer and assemble all maps into an nx1 vector
      kiSizeOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].voOutputSS[ aiB ].viRows   # Get the size of the feature map output
      kiSizeFlat = ( kiSizeOut ** 2 ) * len( aorSelf.voLayersC[ kiCountC - 1 ].voFM )          # Calculate the size of the flattened vector
      aorSelf.voFlatten[ aiB ] = TcMatrix( kiSizeFlat, 1 )                                # Create the flattened vector
      kiF = 0
      for koFM in aorSelf.voLayersC[ kiCountC - 1 ].voFM :          # For each feature map in the last layer
         kdFlat = koFM.voOutputSS[ aiB ].vdData.flatten( )          # Flatten the output of the feature map
         for kiI in range( len( kdFlat ) ) :                        # For each row in the flattened output
            aorSelf.voFlatten[ aiB ].vdData[ kiF ][ 0 ] = kdFlat[ kiI ]
            kiF += 1

      for kiI in range( len( aorSelf.voLayersN ) ) :
         if( kiI == 0 ) :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( aorSelf.voFlatten[ aiB ].vdData, aiB, False )
         else :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( kdRes, aiB, False )

      # Return result
      return( kdRes )

   def MTrain( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
      freeze_support( )
      koPool = Pool( cpu_count( ) )

      for kiEpoch in range( aiEpochs ) :
         # Total Error
         kdError = 0.0

         # Shuffle the input/output pairs
         kdX, kdY = voShuffle( adX, adY, random_state = 0 )

         for kiI in range( 0, len( kdX ), aiBatchSize ) :
            # Pack the Input, Expected Output, and Batch Index
            koArgs = [ ( kdX[ kiI + kiB ], kdY[ kiI + kiB ], kiB ) for kiB in range( aiBatchSize ) ]

            # Execute batch in parallel
            koRes = koPool.map( aorSelf.MLoop, koArgs )
            #koRes = [ ]
            #for kiB in range( aiBatchSize ) :
            #   koRes.append( aorSelf.MLoop( koArgs[ kiB ] ) )
            #kdL = voNP.sum( koRes )

            # Accumulate Error
            kdError += voNP.sum( koRes )

            # Update Kernel Weights and Biases
            aorSelf.MUpdateWeightsBiases( adLR, aiBatchSize )

            # Clear gradients
            aorSelf.MClearGradients( aiBatchSize )

         if( ( kiEpoch % 10 ) == 0 ) :
            adLR /= 2.0 # Reduce Learning Rate
         print( "Epoch: ", kiEpoch, " Error: ",  kdError )

   def MLoop( aorSelf, aoArgs ) :
      kdX = aoArgs[ 0 ] # Parse out Input
      kdY = aoArgs[ 1 ] # Parse out Expected Output
      kiB = aoArgs[ 2 ] # Parse out Batch Index

      # Build input matrix
      koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
      koX.vdData = kdX

      # Run Forward Pass
      kdA = aorSelf.MForwardPass( koX, kiB )

      # Calculate the loss
      kdL = ( ( kdA - kdY ) ** 2 ).sum( )

      # Back Propagate
      aorSelf.MBackPropagate( koX, kdY, kiB )

      # DEBUG PRINT
      # print( "B: ", kiB, "Loss: ", kdL )

      return( kdL )

   def MBackPropagate( aorSelf, aoX, aiY, aiB ) :
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
               for kiJ in range( len( koLayer.vdS[ aiB ] ) ) :
                  if( koLayer.vdS[ aiB ][ kiJ ][ 0 ] < 0 ) :
                     koLayer.vdD[ aiB ][ kiJ ][ 0 ] = 0
               #koLayer.vdD[ aiB ] = 1.0 * ( koLayer.vdS[ aiB ] > 0 )
               #koLayer.vdD[ aiB ] *= ( koLayer.vdS[ aiB ] >= 0 )
            
         else : # Previous Layer
            koLayer.vdD[ aiB ] = voNP.dot( aorSelf.voLayersN[ kiI + 1 ].vdW.T, aorSelf.voLayersN[ kiI + 1 ].vdD[ aiB ] )
            # Apply Dropout
            if( koLayer.vdDropOut < 1.0 ) :
               koLayer.vdD[ aiB ] = voNP.multiply( koLayer.vdD[ aiB ], koLayer.vdDr[ aiB ] )
         
            # Calculate Delta
            if( koLayer.veActivation == TeActivation.XeSigmoid ) :
               koLayer.vdD[ aiB ] = voNP.multiply( koLayer.vdD[ aiB ], koLayer.vdAp[ aiB ] )
            elif( koLayer.veActivation == TeActivation.XeRELU ) :
               for kiJ in range( len( koLayer.vdS[ aiB ] ) ) :
                  if( koLayer.vdS[ aiB ][ kiJ ][ 0 ] < 0 ) :
                     koLayer.vdD[ aiB ][ kiJ ][ 0 ] = 0
               #koLayer.vdD[ aiB ] = 1.0 * ( koLayer.vdS[ aiB ] > 0 )
               #koLayer.vdD[ aiB ] *= ( koLayer.vdS[ aiB ] >= 0 )
         
         koLayer.vdGb[ aiB ] += koLayer.vdD[ aiB ]
            
         if( kiI == 0 ) : # First NN Layer connected to CNN last layer via flatten
            koLayer.vdGw[ aiB ] += voNP.dot( koLayer.vdD[ aiB ], aorSelf.voFlatten[ aiB ].vdData.T  )
         else :
            koLayer.vdGw[ aiB ] += voNP.dot( koLayer.vdD[ aiB ], aorSelf.voLayersN[ kiI - 1].vdA[ aiB ].T ) 
     
      # Compute delta on the output of SS (flat) layer of all feature maps
      kdDss = voNP.dot( aorSelf.voLayersN[ 0 ].vdW.T, aorSelf.voLayersN[ 0 ].vdD[ aiB ] )

      # Reverse flattening and distribute the deltas on each feature map's SS (SubSampling layer)
      koLayer = aorSelf.voLayersC[ kiCountLc - 1 ]
      kiI = 0
      for koFM in koLayer.voFM :
         koFM.voDeltaSS[ aiB ] = TcMatrix( koFM.voOutputSS[ aiB ].viRows, koFM.voOutputSS[ aiB ].viCols )
         for kiM in range( koFM.voOutputSS[ aiB ].viRows ) :
            for kiN in range( koFM.voOutputSS[ aiB ].viCols ) :
               koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] = kdDss[ kiI ]
               kiI += 1
     
      # Process CNN layers in reverse order, from last layer towards input
      for kiI in range( kiCountLc - 1, -1, -1 ) :
         koLayer   = aorSelf.voLayersC[ kiI ]
         kiCountFM = len( koLayer.voFM )

         # Compute deltas on the C layers - distrbute deltas from SS layer then multiply by the activation function
         for kiF in range( kiCountFM ) :
            koFM = koLayer.voFM[ kiF ]
            koFM.voDeltaCV[ aiB ] = TcMatrix( koFM.voOutputSS[ aiB ].viRows * 2, koFM.voOutputSS[ aiB ].viCols * 2 )

            kiIm = 0
            for kiM in range( koFM.voDeltaSS[ aiB ].viRows ) :
               kiIn = 0
               for kiN in range( koFM.voDeltaSS[ aiB ].viCols ) :
                  if( koFM.veActivation == TeActivation.XeSigmoid ) :
                     koFM.voDeltaCV[ aiB ].vdData[ kiIm     ][ kiIn     ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.voAPrime[ aiB ].vdData[ kiIm     ][ kiIn     ]
                     koFM.voDeltaCV[ aiB ].vdData[ kiIm     ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.voAPrime[ aiB ].vdData[ kiIm     ][ kiIn + 1 ]
                     koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn     ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.voAPrime[ aiB ].vdData[ kiIm + 1 ][ kiIn     ]
                     koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.voAPrime[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ]
                  if( koFM.veActivation == TeActivation.XeRELU ) :
                     if( koFM.voSum[ aiB ].vdData[ kiIm ][ kiIn ] > 0 ) :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm ][ kiIn ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm ][ kiIn ] = 0
                     if( koFM.voSum[ aiB ].vdData[ kiIm ][ kiIn + 1 ] > 0 ) :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm ][ kiIn + 1 ] = 0
                     if( koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn ] > 0 ) :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn ] = 0
                     if( koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ] > 0 ) :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ] = 0
                  kiIn = kiIn + 2
               kiIm = kiIm + 2
   
         # Compute Bias Gradients in current CNN Layer
         for koFM in koLayer.voFM :
            for kiU in range( koFM.voDeltaCV[ aiB ].viRows ) :
               for kiV in range( koFM.voDeltaCV[ aiB ].viCols ) :
                  koFM.vdGb[ aiB ] += koFM.voDeltaCV[ aiB ].vdData[ kiU ][ kiV ]
            
         # Compute gradients for pxq kernels in current CNN layer
         if( kiI > 0 ) : # If not the first CNN layer
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koGk = koLayer.voKernelsG[ kiP ][ kiQ ].vdData
                  koMat = koPrev.voFM[ kiP ].voOutputSS[ aiB ].MRotate90( ).MRotate90( )
                  koGk += koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV[ aiB ] ).vdData

            # Backpropagate to prev CNN Layer
            for kiP in range( len( koPrev.voFM ) ) :
               kiSize = koPrev.voFM[ kiP ].voOutputSS[ aiB ].viRows
               koPrev.voFM[ kiP ].voDeltaSS[ aiB ] = TcMatrix( kiSize, kiSize )
               for kiQ in range( len( koLayer.voFM ) ) :
                  koMdss = koPrev.voFM[ kiP ].voDeltaSS[ aiB ].vdData
                  koKernel = koLayer.voKernels[ kiP ][ kiQ ].MRotate90( ).MRotate90( )
                  koMdss += koLayer.voFM[ kiQ ].voDeltaCV[ aiB ].MConvolveFull( koKernel ).vdData
     
         else : # First CNN layer which is connected to input
            # Has 1 x len( voFM ) 2-D array of Kernels and Kernel Gradients
            # Compute gradient for first layer cnn kernels
            for kiQ in range( len( koLayer.voFM ) ) :
               koGk = koLayer.voKernelsG[ 0 ][ kiQ ].vdData
               koMat = aoX.MRotate90( ).MRotate90( )
               koGk += koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV[ aiB ] ).vdData

   def MClearGradients( aorSelf, aiB ) :
      for kiI in range( len( aorSelf.voLayersC ) ) :
         koLayer = aorSelf.voLayersC[ kiI ]

         if( kiI == 0 ) : # First CNN Layer
            for kiQ in range( len( koLayer.voFM ) ) :
               koLayer.voKernelsG[ 0 ][ kiQ ].MClear( )
         else :
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koLayer.voKernelsG[ kiP ][ kiQ ].MClear( );
                
         for koFM in koLayer.voFM :
            koFM.vdGb.fill( 0.0 )
         
      for koLayer in aorSelf.voLayersN :
         koLayer.vdGw.fill( 0 )
         koLayer.vdGb.fill( 0 )

   def MUpdateWeightsBiases( aorSelf, adLR, aiBatchSize ) :
      kiCountLc = len( aorSelf.voLayersC )
      kiCountLn = len( aorSelf.voLayersN )
      
      # Update kernels and weights
      for kiI in range( kiCountLc ) :
         koLayer = aorSelf.voLayersC[ kiI ]

         if( kiI == 0 ) : # First CNN layer
            for kiQ in range( len( koLayer.voFM ) ) :
               koLayer.voKernels[ 0 ][ kiQ ].vdData -= ( koLayer.voKernelsG[ 0 ][ kiQ ].vdData / aiBatchSize ) * adLR 
         else : # Intermediate CNN Layers
            koPrev = aorSelf.voLayersC[ kiI - 1 ]

            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koLayer.voKernels[ kiP ][ kiQ ].vdData -= ( koLayer.voKernelsG[ kiP ][ kiQ ].vdData / aiBatchSize ) * adLR
                
         for koFM in koLayer.voFM :
            koFM.vdBias = koFM.vdBias - ( ( koFM.vdGb.sum( ) / aiBatchSize ) * adLR )

      # Update Regular NN Layers
      for koLayer in aorSelf.voLayersN :
         kdGw = koLayer.vdGw.sum( axis = 0 )
         kdGb = koLayer.vdGb.sum( axis = 0 )

         koLayer.vdW = koLayer.vdW - ( ( kdGw / aiBatchSize ) * adLR )
         koLayer.vdB = koLayer.vdB - ( ( kdGb / aiBatchSize ) * adLR )

