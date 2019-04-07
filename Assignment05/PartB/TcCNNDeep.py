import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle

# Deep Convolutional Neural Network
class TcCNNDeep( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers, aiSizeBatch ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC   = aorLayersC
      aorSelf.voLayersN   = aorLayers
      aorSelf.viSizeBatch = aiSizeBatch
      aorSelf.voFlatten   = voNP.empty( aorSelf.viSizeBatch )

   def MForwardPass( aorSelf, adX, aiI ) :
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
            for kiJ in range( len( aorSelf.voLayersC[ kiI ].voFM ) ) :
               kdPrevOut.append( aorSelf.voLayersC[ kiI ].voFM[ kiJ ].vdOutputSS[ aiI ] )

         # Forward pass on the CNN Layer
         aorSelf.voLayersC[ kiI ].MForwardPass( kdPrevOut, aiI )

      # Flatten each feature map in the CNN Layer and assemble all maps into an nx1 vector
      kiSizeOut  = len( aorSelf.voLayersC[ kiCountC - 1 ].voFM[ 0 ].vdOutputSS[ aiI ] )   # Get the size of the feature map output
      kiSizeFlat = kiSizeOut ** 2                                                         # Calculate the size of the flattened vector
      aorSelf.voFlatten[ aiI ] = voNP.empty( ( kiSizeFlat, 1 ) )                          # Create the flattened vector
      kiF = 0
      for kiI in range( len( aorSelf.voLayerC[ kiCountC - 1 ].voFM ) ) :                     # For each feature map in the last layer
         koOut  = aorSelf.voLayersC[ kiCountC - 1 ].voFM[ kiI ].vdOutputSS[ aiI ]            # Obtain the output of the feature map
         koFlat = koFM.voOutputSS[ aiI ].reshape( koOut.shape[ 0 ] * koOut.shape[ 1 ], 1 )   # Flatten the output of the feature map
         for kiR in range( len( koFlat ) ) :                                                 # For each row in the flattened output
            aorSelf.voFlatten[ aiI ][ kiF ][ 0 ] = koOut[ kiR ][ 0 ]

      # Return result
      return( kdA )

   def MTrain( aorSelf, adX, adY, aiEpochs, adLR, aiBatchSize ) :
      kdA = voNP.zeros( ( aiBatchSize, 1 ) )

      for kiEpoch in range( aiEpochs ) :
         # Zero out the loss
         kdLoss = 0.0

         # Shuffle the input/output pairs
         kdX, kdY = voShuffle( adX, adY, random_state = 0 )

         for kiI in range( 0, len( kdX ), aiBatchSize ) :
            with concurrent.futures.ProcessPoolExecutor() as executor :
               for kiX in range( aiBatchSize ) :
                  kdA[ kiX ] = aorSelf.MForwardPass( kdX[ kiI + kiX ], kiX )


            # Get the Input and Expected Output Batches
            # kdXb = kdX[ kiI : ( kiI + aiBatchSize ) ]
            # kdYb = kdY[ kiI : ( kiI + aiBatchSize ) ]

            # Execute a Foward Pass
            # kdA = aorSelf.MForwardPass( kdXb )

            # Calculate the Loss
            # kdLoss += aorSelf.MLoss( kdA, kdYb )