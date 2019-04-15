import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from multiprocessing import Pool, freeze_support, cpu_count
from TcMatrix import TcMatrix
from TeActivation import TeActivation

# Deep Convolutional Neural Network
class TcCNNDeep( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC   = aorLayersC
      aorSelf.voLayersN   = aorLayers
      aorSelf.voFlatten   = TcMatrix( 1, 1 )
      aorSelf.vdLoss      = 0.0

   def MForwardPass( aorSelf, adX ) :
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
               kdPrevOut.append( koFM.voOutputSS )

         # Forward pass on the CNN Layer
         aorSelf.voLayersC[ kiI ].MForwardPass( kdPrevOut )

      # Flatten each feature map in the CNN Layer and assemble all maps into an nx1 vector
      kiSizeOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].voOutputSS.viRows   # Get the size of the feature map output
      kiSizeFlat = ( kiSizeOut ** 2 ) * len( aorSelf.voLayersC[ kiCountC - 1 ].voFM )          # Calculate the size of the flattened vector
      aorSelf.voFlatten = TcMatrix( kiSizeFlat, 1 )                                # Create the flattened vector
      kiF = 0
      for koFM in aorSelf.voLayersC[ kiCountC - 1 ].voFM :          # For each feature map in the last layer
         kdFlat = koFM.voOutputSS.vdData.flatten( )          # Flatten the output of the feature map
         for kiI in range( len( kdFlat ) ) :                        # For each row in the flattened output
            aorSelf.voFlatten.vdData[ kiF ][ 0 ] = kdFlat[ kiI ]
            kiF += 1

      for kiI in range( len( aorSelf.voLayersN ) ) :
         if( kiI == 0 ) :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( aorSelf.voFlatten.vdData, False )
         else :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( kdRes, False )

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
            koArgs = [ ( kdX[ kiI + kiB ], kdY[ kiI + kiB ] ) for kiB in range( aiBatchSize ) ]

            # Execute batch in parallel
            koRes = koPool.map( aorSelf.MLoop, koArgs )
            #koRes = aorSelf.MLoop( koArgs[ 0 ] )

            # Accumulate Error
            for koCNN in koRes :
               kdError += koCNN.vdLoss

            # Update Kernel Weights and Biases
            aorSelf.MUpdateWeightsBiases( adLR, koRes )

            # Clear gradients
            aorSelf.MClearGradients( )

         if( ( kiEpoch % 10 ) == 0 ) :
            adLR /= 2.0 # Reduce Learning Rate

         print( "Epoch: ", kiEpoch, " Error: ",  kdError )

   def MLoop( aorSelf, aoArgs ) :
      kdX = aoArgs[ 0 ] # Parse out Input
      kdY = aoArgs[ 1 ] # Parse out Expected Output

      # Build input matrix
      koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
      koX.vdData = kdX

      # Run Forward Pass
      kdA = aorSelf.MForwardPass( koX )

      # Calculate the loss
      aorSelf.vdLoss = ( ( kdA - kdY ) ** 2 ).sum( )

      # Back Propagate
      aorSelf.MBackPropagate( koX, kdY )

      return( aorSelf )

   def MBackPropagate( aorSelf, aoX, aiY ) :
      kiCountLn = len( aorSelf.voLayersN )   # Number of NN Layers
      kiCountLc = len( aorSelf.voLayersC )   # Number of CNN Layers

      # Compute Deltas on regular NN layers
      for kiI in range( kiCountLn - 1, -1, -1 ) :  # For each NN Layer, starting with last and working towards first
         koLayer = aorSelf.voLayersN[ kiI ]        # Obtian a reference to the Layer
         if( kiI == ( kiCountLn - 1 ) ) :          # If Last Layer 
            koLayer.vdD = -( aiY - koLayer.vdA )       # Assume SoftMax
            if( koLayer.veActivation == TeActivation.XeSigmoid ) :   # If Activation is Sigmoid
               koLayer.vdD = voNP.multiply( koLayer.vdD, koLayer.vdAp )
            elif( koLayer.veActivation == TeActivation.XeRELU ) :
               for kiJ in range( len( koLayer.vdS ) ) :
                  if( koLayer.vdS[ kiJ ][ 0 ] < 0 ) :
                     koLayer.vdD[ kiJ ][ 0 ] = 0
            
         else : # Previous Layer
            koLayer.vdD = voNP.dot( aorSelf.voLayersN[ kiI + 1 ].vdW.T, aorSelf.voLayersN[ kiI + 1 ].vdD )
            # Apply Dropout
            if( koLayer.vdDropOut < 1.0 ) :
               koLayer.vdD = voNP.multiply( koLayer.vdD, koLayer.vdDr )
         
            # Calculate Delta
            if( koLayer.veActivation == TeActivation.XeSigmoid ) :
               koLayer.vdD = voNP.multiply( koLayer.vdD, koLayer.vdAp )
            elif( koLayer.veActivation == TeActivation.XeRELU ) :
               for kiJ in range( len( koLayer.vdS ) ) :
                  if( koLayer.vdS[ kiJ ][ 0 ] < 0 ) :
                     koLayer.vdD[ kiJ ][ 0 ] = 0
         
         koLayer.vdGb += koLayer.vdD
            
         if( kiI == 0 ) : # First NN Layer connected to CNN last layer via flatten
            koLayer.vdGw += voNP.dot( koLayer.vdD, aorSelf.voFlatten.vdData.T  )
         else :
            koLayer.vdGw += voNP.dot( koLayer.vdD, aorSelf.voLayersN[ kiI - 1].vdA.T ) 
     
      # Compute delta on the output of SS (flat) layer of all feature maps
      kdDss = voNP.dot( aorSelf.voLayersN[ 0 ].vdW.T, aorSelf.voLayersN[ 0 ].vdD )

      # Reverse flattening and distribute the deltas on each feature map's SS (SubSampling layer)
      koLayer = aorSelf.voLayersC[ kiCountLc - 1 ]
      kiI = 0
      for koFM in koLayer.voFM :
         koFM.voDeltaSS = TcMatrix( koFM.voOutputSS.viRows, koFM.voOutputSS.viCols )
         for kiM in range( koFM.voOutputSS.viRows ) :
            for kiN in range( koFM.voOutputSS.viCols ) :
               koFM.voDeltaSS.vdData[ kiM ][ kiN ] = kdDss[ kiI ]
               kiI += 1
     
      # Process CNN layers in reverse order, from last layer towards input
      for kiI in range( kiCountLc - 1, -1, -1 ) :
         koLayer   = aorSelf.voLayersC[ kiI ]
         kiCountFM = len( koLayer.voFM )

         # Compute deltas on the C layers - distrbute deltas from SS layer then multiply by the activation function
         for kiF in range( kiCountFM ) :
            koFM = koLayer.voFM[ kiF ]
            koFM.voDeltaCV = TcMatrix( koFM.voOutputSS.viRows * 2, koFM.voOutputSS.viCols * 2 )

            kiIm = 0
            for kiM in range( koFM.voDeltaSS.viRows ) :
               kiIn = 0
               for kiN in range( koFM.voDeltaSS.viCols ) :
                  if( koFM.veActivation == TeActivation.XeSigmoid ) :
                     koFM.voDeltaCV.vdData[ kiIm     ][ kiIn     ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ] * koFM.voAPrime.vdData[ kiIm     ][ kiIn     ]
                     koFM.voDeltaCV.vdData[ kiIm     ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ] * koFM.voAPrime.vdData[ kiIm     ][ kiIn + 1 ]
                     koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn     ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ] * koFM.voAPrime.vdData[ kiIm + 1 ][ kiIn     ]
                     koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ] * koFM.voAPrime.vdData[ kiIm + 1 ][ kiIn + 1 ]
                  if( koFM.veActivation == TeActivation.XeRELU ) :
                     if( koFM.voSum.vdData[ kiIm ][ kiIn ] > 0 ) :
                        koFM.voDeltaCV.vdData[ kiIm ][ kiIn ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV.vdData[ kiIm ][ kiIn ] = 0
                     if( koFM.voSum.vdData[ kiIm ][ kiIn + 1 ] > 0 ) :
                        koFM.voDeltaCV.vdData[ kiIm ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV.vdData[ kiIm ][ kiIn + 1 ] = 0
                     if( koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn ] > 0 ) :
                        koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn ] = 0
                     if( koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn + 1 ] > 0 ) :
                        koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS.vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV.vdData[ kiIm + 1 ][ kiIn + 1 ] = 0
                  kiIn = kiIn + 2
               kiIm = kiIm + 2
   
         # Compute Bias Gradients in current CNN Layer
         for koFM in koLayer.voFM :
            for kiU in range( koFM.voDeltaCV.viRows ) :
               for kiV in range( koFM.voDeltaCV.viCols ) :
                  koFM.vdGb += koFM.voDeltaCV.vdData[ kiU ][ kiV ]
            
         # Compute gradients for pxq kernels in current CNN layer
         if( kiI > 0 ) : # If not the first CNN layer
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koGk = koLayer.voKernelsG[ kiP ][ kiQ ].vdData
                  koMat = koPrev.voFM[ kiP ].voOutputSS.MRotate90( ).MRotate90( )
                  koGk += koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV ).vdData

            # Backpropagate to prev CNN Layer
            for kiP in range( len( koPrev.voFM ) ) :
               kiSize = koPrev.voFM[ kiP ].voOutputSS.viRows
               koPrev.voFM[ kiP ].voDeltaSS = TcMatrix( kiSize, kiSize )
               for kiQ in range( len( koLayer.voFM ) ) :
                  koMdss = koPrev.voFM[ kiP ].voDeltaSS.vdData
                  koKernel = koLayer.voKernels[ kiP ][ kiQ ].MRotate90( ).MRotate90( )
                  koMdss += koLayer.voFM[ kiQ ].voDeltaCV.MConvolveFull( koKernel ).vdData
     
         else : # First CNN layer which is connected to input
            # Has 1 x len( voFM ) 2-D array of Kernels and Kernel Gradients
            # Compute gradient for first layer cnn kernels
            for kiQ in range( len( koLayer.voFM ) ) :
               koGk = koLayer.voKernelsG[ 0 ][ kiQ ].vdData
               koMat = aoX.MRotate90( ).MRotate90( )
               koGk += koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV ).vdData

   def MClearGradients( aorSelf ) :
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
            koFM.vdGb = 0.0
         
      for koLayer in aorSelf.voLayersN :
         koLayer.vdGw.fill( 0 )
         koLayer.vdGb.fill( 0 )

   def MUpdateWeightsBiases( aorSelf, adLR, aoCNN ) :
      kiCountLc = len( aorSelf.voLayersC )   # Number of CNN Layers
      kiCountLn = len( aorSelf.voLayersN )   # Number of NN Layers
      kiSizeB   = len( aoCNN )               # Batch Size
      
      # Update kernels and weights
      for kiI in range( kiCountLc ) :
         koLayer = aorSelf.voLayersC[ kiI ]

         kdGk = voNP.zeros( ( koLayer.voKernelsG[ 0 ][ 0 ].viRows, koLayer.voKernelsG[ 0 ][ 0 ].viCols ) ) 
         if( kiI == 0 ) : # First CNN layer
            for kiQ in range( len( koLayer.voFM ) ) :
               kdGk.fill( 0 )
               for kiB in range( kiSizeB ) :
                  kdGk += aoCNN[ kiB ].voLayersC[ kiI ].voKernelsG[ 0 ][ kiQ ].vdData
               koLayer.voKernels[ 0 ][ kiQ ].vdData -= ( kdGk / kiSizeB ) * adLR 
         else : # Intermediate CNN Layers
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  kdGk.fill( 0 )
                  for kiB in range( kiSizeB ) :
                     kdGk += aoCNN[ kiB ].voLayersC[ kiI ].voKernelsG[ kiP ][ kiQ ].vdData
                  koLayer.voKernels[ kiP ][ kiQ ].vdData -= ( kdGk / kiSizeB ) * adLR
                
         for kiF in range( len( koLayer.voFM ) ) :
            koFM = koLayer.voFM[ kiF ]
            kdGb = koFM.vdGb
            for kiB in range( kiSizeB ) :
               kdGb += aoCNN[ kiB ].voLayersC[ kiI ].voFM[ kiF ].vdGb
            koFM.vdBias = koFM.vdBias - ( ( kdGb / kiSizeB ) * adLR )

      # Update Regular NN Layers
      for kiI in range( kiCountLn ) :
         koLayer = aorSelf.voLayersN[ kiI ]
         kdGw = koLayer.vdGw
         kdGb = koLayer.vdGb

         for kiB in range( kiSizeB ) :
            kdGw += aoCNN[ kiB ].voLayersN[ kiI ].vdGw
            kdGb += aoCNN[ kiB ].voLayersN[ kiI ].vdGb

         koLayer.vdW = koLayer.vdW - ( ( kdGw / kiSizeB ) * adLR )
         koLayer.vdB = koLayer.vdB - ( ( kdGb / kiSizeB ) * adLR )

