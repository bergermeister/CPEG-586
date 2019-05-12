import concurrent.futures
import numpy as voNP
import copy
from sklearn.utils import shuffle as voShuffle
from multiprocessing import Pool, freeze_support, cpu_count
from TcMatrix import TcMatrix
from TeActivation import TeActivation

# Deep Convolutional Neural Network
class TcCNNSiamese( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC   = aorLayersC
      aorSelf.voLayersN   = aorLayers
      aorSelf.voFlatten   = TcMatrix( 1, 1 )
      aorSelf.vdLossContr = 0.0  # Contrastive Loss
      aorSelf.vdLossCross = 0.0  # Cross Entropy Loss

   def MNetworkModel( aorSelf, adX ) :
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
      kiSizeOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].voOutputSS.viRows       # Get the size of the feature map output
      kiSizeFlat = ( kiSizeOut ** 2 ) * len( aorSelf.voLayersC[ kiCountC - 1 ].voFM )  # Calculate the size of the flattened vector
      aorSelf.voFlatten = TcMatrix( kiSizeFlat, 1 )                                    # Create the flattened vector
      kiF = 0
      for koFM in aorSelf.voLayersC[ kiCountC - 1 ].voFM :                             # For each feature map in the last layer
         kdFlat = koFM.voOutputSS.vdData.flatten( )                                    # Flatten the output of the feature map
         for kiI in range( len( kdFlat ) ) :                                           # For each row in the flattened output
            aorSelf.voFlatten.vdData[ kiF ][ 0 ] = kdFlat[ kiI ]
            kiF += 1

      return( aorSelf.voFlatten )

   def MNetworkClassifier( aorSelf, adX ) :
      koFlatten = aorSelf.MNetworkModel( adX )

      for kiI in range( len( aorSelf.voLayersN ) ) :
         if( kiI == 0 ) :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( koFlatten.vdData, False )
         else :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( kdRes, False )

      # Return result
      return( kdRes )
   
   def MTrainModel( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
      freeze_support( )
      koPool = Pool( cpu_count( ) )

      # Train the network for Modeldings via contrastive loss
      for kiE in range( aiEpochs ) :
         # Total Error
         kdError = 0.0     

         # Shuffle Input/Output pairs
         kdX1, kdY1 = voShuffle( adX, adY, random_state = 0 )
         kdX2, kdY2 = voShuffle( adX, adY, random_state = 10 )

         for kiI in range( 0, len( kdX1 ), aiBatchSize ) :
            # Pack the Intp
            koArgs = [ ( kdX1[ kiI + kiB ], kdX2[ kiI + kiB ], voNP.argmax( kdY1[ kiI + kiB ] ) != voNP.argmax( kdY2[ kiI + kiB ] ) ) for kiB in range( aiBatchSize ) ]
            
            # Execute batch in parallel
            koRes = koPool.map( aorSelf.MLoopModel, koArgs )
            #koRes = [ ]
            #for kiB in range( aiBatchSize ) :
            #   koRes.append( aorSelf.MLoopModel( koArgs[ kiB ] ) )

            # Accumulate Error
            for koCNN in koRes :
               kdError += koCNN.vdLossContr

            # Back Propagate
            aorSelf.MBackPropagateModel( adLR, koRes )

            # Clear gradients
            aorSelf.MClearGradients( )

         if( ( kiE % 10 ) == 0 ) :
            adLR /= 2.0 # Reduce Learning Rate

         print( "Epoch: ", kiE, " Error: ",  kdError )

   def MTrainClassifier( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
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
            koRes = koPool.map( aorSelf.MLoopClassifier, koArgs )

            # Accumulate Error
            for koCNN in koRes :
               kdError += koCNN.vdLossCross

            # Update Kernel Weights and Biases
            aorSelf.MBackPropagateClassifier( adLR, koRes )

            # Clear gradients
            aorSelf.MClearGradients( )

         if( ( kiEpoch % 10 ) == 0 ) :
            adLR /= 2.0 # Reduce Learning Rate

         print( "Epoch: ", kiEpoch, " Error: ",  kdError )

   def MLoopModel( aorSelf, aoArgs ) :
      kdX1 = aoArgs[ 0 ] # Parse Input 1
      kdX2 = aoArgs[ 1 ] # Parse Input 2
      kdY  = aoArgs[ 2 ] # Parse Expected Output 

      kdMargin = 5.0 #0.2

      # Build Input Matrices
      koX1 = TcMatrix( kdX1.shape[ 0 ], kdX1.shape[ 1 ] )
      koX1.vdData = kdX1
      koX2 = TcMatrix( kdX2.shape[ 0 ], kdX2.shape[ 1 ] )
      koX2.vdData = kdX2

      # Run Forward Pass for CNN with Input X1 and copy the results
      kdA1 = aorSelf.MNetworkModel( koX1 )
      koN1 = copy.deepcopy( aorSelf )

      # Run Forward Pass for CNN with Input X2 and copy the results
      kdA2 = aorSelf.MNetworkModel( koX2 )
      koN2 = copy.deepcopy( aorSelf )

      # Calculate the contrastive loss
      kdDw = voNP.sqrt( voNP.maximum( 1e-6, voNP.sum( ( kdA1.vdData - kdA2.vdData ) ** 2 ) ) )
      kdLs = 0.5 * voNP.multiply( ( 1 - kdY ), ( kdDw ** 2 ) )
      kdLd = 0.5 * voNP.multiply( kdY, voNP.maximum( 0.0, kdMargin - kdDw ) ** 2 )
      aorSelf.vdLossContr = voNP.mean( kdLs + kdLd )

      # Calculate Delta and apply derivate of RELU
      kdD = voNP.sum( ( kdA1.vdData - kdA2.vdData ), 1 )
      #kdD = kdD * voNP.sum( voNP.maximum( 0, kdA1.vdData ) - voNP.maximum( 0, kdA2.vdData ) )
      if( kdY == True ) :
         if( kdDw < kdMargin ) :
            kdD = kdD * -( ( kdMargin - kdDw ) / kdDw )
         else :
            kdD = kdD * 0.0

      # Update Gradients
      koN1.MUpdateModel( koX1, kdD, kdY )
      koN2.MUpdateModel( koX2, kdD, kdY )
      for kiI in range( len( aorSelf.voLayersC ) ) :
         koLayer = aorSelf.voLayersC[ kiI ]
         koL1    = koN1.voLayersC[ kiI ]
         koL2    = koN2.voLayersC[ kiI ]
         if( kiI == 0 ) : # First CNN layer
            for kiQ in range( len( koLayer.voFM ) ) :
               koLayer.voKernelsG[ 0 ][ kiQ ].vdData = koL1.voKernelsG[ 0 ][ kiQ ].vdData - koL2.voKernelsG[ 0 ][ kiQ ].vdData
         else : # Intermediate CNN Layers
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koLayer.voKernelsG[ kiP ][ kiQ ].vdData = koL1.voKernelsG[ kiP ][ kiQ ].vdData - koL2.voKernelsG[ kiP ][ kiQ ].vdData

      return( aorSelf )

   def MLoopClassifier( aorSelf, aoArgs ) :
      kdX = aoArgs[ 0 ] # Parse out Input
      kdY = aoArgs[ 1 ] # Parse out Expected Output

      # Build input matrix
      koX = TcMatrix( kdX.shape[ 0 ], kdX.shape[ 1 ] )
      koX.vdData = kdX

      # Run Forward Pass
      kdA = aorSelf.MNetworkClassifier( koX )

      # Calculate the loss
      aorSelf.vdLossCross = ( ( kdA - kdY ) ** 2 ).sum( )

      # Back Propagate
      aorSelf.MUpdateClassifier( koX, kdY )

      return( aorSelf )

   def MUpdateModel( aorSelf, aoX, adDw, aiY ) :
      kiCountLc = len( aorSelf.voLayersC ) # Number of CNN Layers

      # Reverse flattening and distribute the deltas on each feature map's SS (SubSampling layer)
      koLayer = aorSelf.voLayersC[ kiCountLc - 1 ]
      kiI = 0
      for koFM in koLayer.voFM :
         koFM.voDeltaSS = TcMatrix( koFM.voOutputSS.viRows, koFM.voOutputSS.viCols )
         for kiM in range( koFM.voOutputSS.viRows ) :
            for kiN in range( koFM.voOutputSS.viCols ) :
               koFM.voDeltaSS.vdData[ kiM ][ kiN ] = adDw[ kiI ]
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
                  koGk += koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV, 'valid' ).vdData

            # Backpropagate to prev CNN Layer
            for kiP in range( len( koPrev.voFM ) ) :
               kiWidth = koPrev.voFM[ kiP ].voOutputSS.viRows
               kiHeight = koPrev.voFM[ kiP ].voOutputSS.viCols
               koPrev.voFM[ kiP ].voDeltaSS = TcMatrix( kiWidth, kiHeight )
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

   def MUpdateClassifier( aorSelf, aoX, aiY ) :
      kiCountLn = len( aorSelf.voLayersN )   # Number of NN Layers

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

   def MBackPropagateModel( aorSelf, adLR, aoCNN ) :
      kiCountLc = len( aorSelf.voLayersC )   # Number of CNN Layers
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

   def MBackPropagateClassifier( aorSelf, adLR, aoCNN ) :
      kiCountLn = len( aorSelf.voLayersN )   # Number of NN Layers
      kiSizeB   = len( aoCNN )               # Batch Size

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