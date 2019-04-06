import concurrent.futures
import numpy as voNP
from sklearn.utils import shuffle as voShuffle

# Deep Convolutional Neural Network
class TcCNNDeep( object ) :
   def __init__( aorSelf, aorLayersC, aorLayers ) :
      # Save the CNN Layers and NN Layers
      aorSelf.voLayersC = aorLayersC
      aorSelf.voLayersN = aorLayers

   def MForwardPass( aorSelf, adX, aiIndex ) :
      # Initialize empty list of results
      kdA = []

      # Execute first CNN Layer
      kdA.append( aorSelf.voLayersC[ 0 ].MForwardPass( adX ) )

      # Execute each intermediate CNN Layer
      for kiI in range( 1, len( aorSelf.voLayersC ) ) :
         kdA.append( aorSelf.voLayersC[ kiI ].MForwardPass( kdA[ kiI - 1 ] ) )

      # Run Forward Pass on first layer
      kdA = aorSelf.voLayer[ 0 ].MForwardPass( adX )

      # Run Forward Pass on Hidden Layers and Last Layer
      for kiI in range( 1, len( aorSelf.voLayer ) ) :
         kdA = aorSelf.voLayer[ kiI ].MForwardPass( kdA )

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
            with concurrent.futures.ProcessPoolExecutor() as executor:
               for kiX in range( aiBatchSize ) :
                  kdA[ kiX ] = aorSelf.MForwardPass( kdXb[ kiI + kiX ] )


            # Get the Input and Expected Output Batches
            # kdXb = kdX[ kiI : ( kiI + aiBatchSize ) ]
            # kdYb = kdY[ kiI : ( kiI + aiBatchSize ) ]

            # Execute a Foward Pass
            # kdA = aorSelf.MForwardPass( kdXb )

            # Calculate the Loss
            # kdLoss += aorSelf.MLoss( kdA, kdYb )