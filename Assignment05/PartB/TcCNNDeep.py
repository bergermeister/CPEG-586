import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from multiprocessing import Process
import multiprocessing
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
      aorSelf.vdError     = 0.0

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
               kdPrevOut.append( aorSelf.voLayersC[ kiI - 1 ].voFM[ kiJ ].voOutputSS[ aiB ] )

         # Forward pass on the CNN Layer
         aorSelf.voLayersC[ kiI ].MForwardPass( kdPrevOut, aiB )

      # Flatten each feature map in the CNN Layer and assemble all maps into an nx1 vector
      kiSizeOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].voOutputSS[ aiB ].viRows   # Get the size of the feature map output
      kiSizeFlat = ( kiSizeOut ** 2 ) * len( aorSelf.voLayersC[ kiCountC - 1 ].voFM )          # Calculate the size of the flattened vector
      aorSelf.voFlatten[ aiB ] = TcMatrix( kiSizeFlat, 1 )                                # Create the flattened vector
      kiF = 0
      for kiI in range( len( aorSelf.voLayersC[ kiCountC - 1 ].voFM ) ) :              # For each feature map in the last layer
         koOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ kiI ].voOutputSS[ aiB ] # Obtain the output of the feature map
         kdFlat = koOut.vdData.flatten( )                                              # Flatten the output of the feature map
         for kiR in range( len( kdFlat ) ) :                                           # For each row in the flattened output
            aorSelf.voFlatten[ aiB ].vdData[ kiF ] = kdFlat[ kiR ]
            kiF += 1

      for kiI in range( len( aorSelf.voLayersN ) ) :
         if( kiI == 0 ) :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( aorSelf.voFlatten[ aiB ].vdData, aiB, False )
         else :
            kdRes = aorSelf.voLayersN[ kiI ].MForwardPass( kdRes, aiB, False )

      # Return result
      return( kdRes )

   def MTrain( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
      koProc = voNP.ndarray( aiBatchSize, dtype=Process )

      for kiEpoch in range( aiEpochs ) :
         # Total Error
         aorSelf.vdError = 0.0

         # Shuffle the input/output pairs
         kdX, kdY = voShuffle( adX, adY, random_state = 0 )

         for kiI in range( 0, len( kdX ), aiBatchSize ) :
            
            for kiB in range( aiBatchSize ) :
               kiJ = kiI + kiB
               #aorSelf.MLoop( kdX[ kiJ ], kdY[ kiJ ], kiB )
               koProc[ kiB ] = Process( target=aorSelf.MLoop, args=( kdX[ kiJ ], kdY[ kiJ ], kiB ) )
               koProc[ kiB ].start( )
            
            for kiB in range( aiBatchSize ) :
               koProc[ kiB ].join( )

            # with concurrent.futures.ProcessPoolExecutor() as executor :
            #    for kiB in range( aiBatchSize ) :
            #       kiJ = kiI + kiB
            #       koX = TcMatrix( kdX[ kiJ ].shape[ 0 ], kdX[ kiJ ].shape[ 1 ] )
            #       koX.vdData = kdX[ kiJ ]
            #       kdA = aorSelf.MForwardPass( koX, kiB )
            # 
            #       # Calculate the loss
            #       kdL = ( ( kdA - kdY[ kiJ ] ) ** 2 )
            #       
            #       # Accumulate Error
            #       kdError += kdL.sum( )
            # 
            #       aorSelf.MBackPropagate( koX, kdY[ kiJ ], kiB )

            # Update Kernel Weights and Biases
            aorSelf.MUpdateWeightsBiases( adLR, aiBatchSize )

            # Clear gradients
            aorSelf.MClearGradients( aiBatchSize )

         print( "Epoch: ", kiEpoch, " Error: ",  aorSelf.vdError )

   def MLoop( aorSelf, adX, adY, aiB ) :
      koX = TcMatrix( adX.shape[ 0 ], adX.shape[ 1 ] )
      koX.vdData = adX
      kdA = aorSelf.MForwardPass( koX, aiB )

      # Calculate the loss
      kdL = ( ( kdA - adY ) ** 2 )
                  
      # Accumulate Error
      aorSelf.vdError += kdL.sum( )

      aorSelf.MBackPropagate( koX, adY, aiB )

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
            koLayer.vdGw[ aiB ] = koLayer.vdGw[ aiB ] + voNP.dot( koLayer.vdD[ aiB ], aorSelf.voFlatten[ aiB ].vdData.T  )
         else :
            koLayer.vdGw[ aiB ] = koLayer.vdGw[ aiB ] + voNP.dot( koLayer.vdD[ aiB ], aorSelf.voLayersN[ kiI - 1].vdA[ aiB ].T ) 
     
      # Compute delta on the output of SS (flat) layer of all feature maps
      kdDss = voNP.dot( aorSelf.voLayersN[ 0 ].vdW.T, aorSelf.voLayersN[ 0 ].vdD[ aiB ] )

      # Reverse flattening and distribute the deltas on each feature map's SS (SubSampling layer)
      koLayer = aorSelf.voLayersC[ kiCountLc - 1 ]
      kiI = 0
      for kiF in range( len( koLayer.voFM ) ) :
         koFM = koLayer.voFM[ kiF ]
         koFM.voDeltaSS[ aiB ] = TcMatrix( koFM.voOutputSS[ aiB ].viRows, koFM.voOutputSS[ aiB ].viCols )
         for kiR in range( koFM.voOutputSS[ aiB ].viRows ) :
            for kiC in range( koFM.voOutputSS[ aiB ].viCols ) :
               koFM.voDeltaSS[ aiB ].vdData[ kiR ][ kiC ] = kdDss[ kiI ]
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
                     koFM.voDeltaCV[ aiB ].vdData[ aiIm     ][ aiIn     ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.APrime[ aiB ].vdData[ kiIm     ][ kiIn     ]
                     koFM.voDeltaCV[ aiB ].vdData[ aiIm     ][ aiIn + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.APrime[ aiB ].vdData[ kiIm     ][ kiIn + 1 ]
                     koFM.voDeltaCV[ aiB ].vdData[ aiIm + 1 ][ aiN      ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.APrime[ aiB ].vdData[ kiIm + 1 ][ kiIn     ]
                     koFM.voDeltaCV[ aiB ].vdData[ aiIm + 1 ][ aiN  + 1 ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ] * koFM.APrime[ aiB ].vdData[ kiIm + 1 ][ kiIn + 1 ]
                  if( koFM.veActivation == TeActivation.XeRELU ) :
                     if( koFM.voSum[ aiB ].vdData[ kiIm ][ kiIn ] > 0 ) :
                        koFM.voDeltaCV[ aiB ].vdData[ kiM ][ kiIn ] = ( 1 / 4.0 ) * koFM.voDeltaSS[ aiB ].vdData[ kiM ][ kiN ]
                     else :
                        koFM.voDeltaCV[ aiB ].vdData[ kiM ][ kiIn ] = 0
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
         for kiF in range( kiCountFM ) :
            koFM = koLayer.voFM[ kiF ]
            for kiU in range( koFM.voDeltaCV[ aiB ].viRows ) :
               for kiV in range( koFM.voDeltaCV[ aiB ].viCols ) :
                  koFM.vdGb += koFM.voDeltaCV[ aiB ].vdData[ kiU ][ kiV ]
            
         # Compute gradients for pxq kernels in current CNN layer
         if( kiI > 0 ) : # If not the first CNN layer
            koPrev = aorSelf.voLayersC[ kiI - 1 ]
            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koMat = koPrev.voFM[ kiP ].voOutputSS[ aiB ].MRotate90( ).MRotate90( )
                  koLayer.voKernelsG[ kiP ][ kiQ ].vdData = koLayer.voKernelsG[ kiP ][ kiQ ].vdData + koMat.MConvolve( koLayer.voFM[ kiQ ].voDeltaCV[ aiB ] ).vdData

            # Backpropagate to prev CNN Layer
            for kiP in range( len( koPrev.voFM ) ) :
               kiSize = koPrev.voFM[ kiP ].voOutputSS[ aiB ].viRows
               koPrev.voFM[ kiP ].voDeltaSS[ aiB ] = TcMatrix( kiSize, kiSize )
               for kiQ in range( len( koLayer.voFM ) ) :
                  koPrev.voFM[ kiP ].voDeltaSS[ aiB ].vdData = koPrev.voFM[ kiP ].voDeltaSS[ aiB ].vdData + \
                                                               koLayer.voFM[ kiQ ].voDeltaCV[ aiB ].MConvolveFull( \
                                                                  koLayer.voKernels[ kiP ][ kiQ ].MRotate90( ).MRotate90( ) ).vdData
     
         else : # First CNN layer which is connected to input
            # Has 1 x len( voFM ) 2-D array of Kernels and Kernel Gradients
            # Compute gradient for first layer cnn kernels
            for kiQ in range( len( koLayer.voFM ) ) :
               koLayer.voKernelsG[ 0 ][ kiQ ].vdData = koLayer.voKernelsG[ 0 ][ kiQ ].vdData + aoX.MRotate90( ).MRotate90( ).MConvolve( koLayer.voFM[ kiQ ].voDeltaCV[ aiB ] ).vdData


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
                
         for kiF in range( len( koLayer.voFM ) ) :
            koLayer.voFM[ kiF ].vdGb = 0.0
         
      for kiI in range( len( aorSelf.voLayersN ) ) :
         koLayer = aorSelf.voLayersN[ kiI ]
         koLayer.vdGw = voNP.zeros( koLayer.vdGw.shape )
         koLayer.vdGb = voNP.zeros( koLayer.vdGb.shape )

   def MUpdateWeightsBiases( aorSelf, adLR, aiBatchSize ) :
      kiCountLc = len( aorSelf.voLayersC )
      kiCountLn = len( aorSelf.voLayersN )
      
      # Update kernels and weights
      for kiI in range( kiCountLc ) :
         koLayer = aorSelf.voLayersC[ kiI ]

         if( kiI == 0 ) : # First CNN layer
            for kiQ in range( len( koLayer.voFM ) ) :
               koLayer.voKernels[ 0 ][ kiQ ].vdData = koLayer.voKernels[ 0 ][ kiQ ].vdData - \
                                                      voNP.multiply( koLayer.voKernelsG[ 0 ][ kiQ ].vdData, adLR * ( 1.0 / aiBatchSize ) )
         else : # Intermediate CNN Layers
            koPrev = aorSelf.voLayersC[ kiI - 1 ]

            for kiP in range( len( koPrev.voFM ) ) :
               for kiQ in range( len( koLayer.voFM ) ) :
                  koLayer.voKernels[ kiP ][ kiQ ].vdData = koLayer.voKernels[ kiP ][ kiQ ].vdData - \
                                                           voNP.multiply( koLayer.voKernelsG[ kiP ][ kiQ ].vdData, adLR * ( 1.0 / aiBatchSize ) )
                
         for kiF in range( len( koLayer.voFM ) ) :
            koFM = koLayer.voFM[ kiF ]
            koFM.vdBias = koFM.vdBias - ( ( koFM.vdGb / aiBatchSize ) * adLR )

      # Update Regular NN Layers
      for kiI in range( kiCountLn ) :
         koLayer = aorSelf.voLayersN[ kiI ]
         kdGw = koLayer.vdGw.sum( axis = 0 )
         kdGb = koLayer.vdGb.sum( axis = 0 )

         koLayer.vdW = koLayer.vdW - ( kdGw * ( 1.0 / aiBatchSize ) * adLR )
         koLayer.vdB = koLayer.vdB - ( kdGb * ( 1.0 / aiBatchSize ) * adLR )

