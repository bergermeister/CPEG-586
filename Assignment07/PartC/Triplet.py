import os
import tensorflow as voTF
import numpy as voNP
from keras.utils import to_categorical
from sklearn.utils import shuffle as voShuffle
from Activation import TeActivation
from Pool import TePool

class TcTriplet( object ) :
   def __init__( aorSelf ) :
      # Set up place holders for inputs
      aorSelf.voInRef = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InRef' ) # Ref image input
      aorSelf.voInPos = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InPos' ) # Same image input
      aorSelf.voInNeg = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InNeg' ) # Diff image input

      # Set up place holders for labels
      aorSelf.voY = voTF.placeholder( voTF.float32, [ None, ], name = 'voY' )
      aorSelf.voYo = voTF.placeholder( voTF.float32, [ None, 10 ], name = 'YOneHot' )

      # Setup the Outputs, Loss Function, and Training MNetworkTriplet
      aorSelf.voOutRef, aorSelf.voOutPos, aorSelf.voOutNeg = aorSelf.MNetworkTriplet( )
      aorSelf.voOutput = aorSelf.MNetworkClassifier( )
      aorSelf.vdLossTriplet = aorSelf.MLossTriplet( )
      aorSelf.vdLossCrossE  = aorSelf.MLossCrossEntropy( )
      aorSelf.voOptimizerTriplet = aorSelf.MOptimizerTriplet( )
      aorSelf.voOptimizerCrossE  = aorSelf.MOptimizerCrossE( )
      aorSelf.voSaver = voTF.train.Saver( )

      # Initialize TensorFlow session
      aorSelf.voSession = voTF.Session( )
      aorSelf.voSession.run( voTF.global_variables_initializer( ) )

   def MLayerD( aorSelf, aoX, aoShape, aeActivation, aoVarName, abTrainable = True ) :
      # Initialize weights and bias
      koW = voTF.get_variable( name = aoVarName + '_W', 
                               dtype = voTF.float32, 
                               shape = aoShape,
                               initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                               trainable = abTrainable )
      koB = voTF.get_variable( name = aoVarName + '_B', 
                               dtype = voTF.float32, 
                               shape = [ aoShape[ 1 ] ],
                               initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                               trainable = abTrainable )

      # Apply weights and bias to input
      koY = voTF.add( voTF.matmul( aoX, koW ), koB )

      # Apply activation function
      if( aeActivation == TeActivation.XeSigmoid ) :
         koY = voTF.nn.sigmoid( koY )
      elif( aeActivation == TeActivation.XeRELU ) :
         koY = voTF.nn.relu( koY )
      elif( aeActivation == TeActivation.XeSoftMax ) :
         koY = voTF.nn.softmax( koY )

      return( koY )

   def MLayerC( aorSelf, aoX, aoShapeLayer, aoShapeKernel, aoShapePool, aeActivation, aePool, aoVarName, abTrainable = True ) :
      # Setup Shapes: Kernel Rows           Kernel Cols         Input              Num Kernels
      koShapeKernel = [ aoShapeKernel[ 0 ], aoShapeKernel[ 1 ], aoShapeLayer[ 0 ], aoShapeLayer[ 1 ] ]

      # Initialise weights and bias for the kernel
      koW = voTF.get_variable( name = aoVarName + '_W',
                              dtype = voTF.float32,
                              shape = koShapeKernel,
                              initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                              trainable = abTrainable )
      koB = voTF.get_variable( name = aoVarName + '_B',
                              dtype = voTF.float32,
                              shape = aoShapeLayer[ 1 ],
                              initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                              trainable = abTrainable )

      # Perform Convolution
      koY = voTF.nn.conv2d( aoX, koW, [ 1, 1, 1, 1 ], padding = 'SAME' )

      # Add the bias
      koY += koB

      # Apply activation function
      if( aeActivation == TeActivation.XeSigmoid ) :
         koY = voTF.nn.sigmoid( koY )
      elif( aeActivation == TeActivation.XeRELU ) :
         koY = voTF.nn.relu( koY )

      # Apply Pooling
      koSize = [ 1, aoShapePool[ 0 ], aoShapePool[ 1 ], 1 ]
      koStride = [ 1, aoShapePool[ 0 ], aoShapePool[ 1 ], 1 ]
      if( aePool == TePool.XeAvg ) :
         koY = voTF.nn.avg_pool( koY, ksize = koSize, strides = koStride, padding = 'SAME' )
      elif( aePool == TePool.XeMax ) :
         koY = voTF.nn.max_pool( koY, ksize = koSize, strides = koStride, padding = 'SAME' )

      return( koY )

   def MCNN( aorSelf, aoInput, abTrainable = True ) :
      # Reshape the input
      koX = voTF.reshape( aoInput, [ -1, 28, 28, 1 ] )

      # Setup Convolutional Layers
      koLC1 = aorSelf.MLayerC( koX,   [  1,   32 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC1' )
      koLC2 = aorSelf.MLayerC( koLC1, [ 32,   64 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC2' )
      koLC3 = aorSelf.MLayerC( koLC2, [ 64,  128 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeNone, TePool.XeMax, 'LC3' )

      # Flatten the output
      #koFlat = voTF.reshape( koLC2, [ -1, 7 * 7 * 64 ] )
      koFlat = voTF.reshape( koLC3, [ -1, 4 * 4 * 128 ] )

      #koX = aoInput
      #koLE1 = aorSelf.MLayerD( koX,   [ 784, 500 ], TeActivation.XeRELU, 'LE1', abTrainable )
      #koLE2 = aorSelf.MLayerD( koLE1, [ 500, 250 ], TeActivation.XeRELU, 'LE2', abTrainable )
      #koLE3 = aorSelf.MLayerD( koLE2, [ 250, 128 ], TeActivation.XeNone, 'LE3', abTrainable )
      #koFlat = koLE3

      return( koFlat )

   def MNetworkTriplet( aorSelf ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         koOutRef = aorSelf.MCNN( aorSelf.voInRef, True )
         
         # Share weights
         koScope.reuse_variables( )
         koOutPos = aorSelf.MCNN( aorSelf.voInPos, True )
         koOutNeg = aorSelf.MCNN( aorSelf.voInNeg, True )

      return( koOutRef, koOutPos, koOutNeg )

   def MNetworkClassifier( aorSelf ) :
      # Initialize Neural Network
      with voTF.variable_scope( "triplet", reuse = voTF.AUTO_REUSE ) as koScope :
         koCNN = aorSelf.MCNN( aorSelf.voInRef, abTrainable = False )
         koA1  = voTF.nn.relu( koCNN )

         # Setup Dense Layers
         koLD1 = aorSelf.MLayerD( koA1, [ koA1.get_shape( )[ 1 ], 100 ], TeActivation.XeRELU, 'LD1' )
         koLD2 = aorSelf.MLayerD( koLD1,  [ 100, 10 ], TeActivation.XeSoftMax, 'LD2' )

      return( koLD2 )

   def MLossTriplet( aorSelf ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         kdMargin = 0.2
         koX = aorSelf.voOutRef
         koXp = aorSelf.voOutPos
         koXn = aorSelf.voOutNeg

         koDp2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( koX, koXp ) ), 1 )
         koDn2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( koX, koXn ) ), 1 )

         #koDp2 = voTF.nn.softmax( koDp2 )
         #koDn2 = voTF.nn.softmax( koDn2 )

         #koLoss = voTF.math.divide( voTF.math.add( koDp2, adMargin ), voTF.math.add( koDn2, 1e-6 ) )
         koLoss = voTF.maximum( 0.0, voTF.math.add( voTF.math.subtract( koDp2, koDn2 ), kdMargin ) )

      return( voTF.reduce_mean( koLoss ) ) #, voTF.reduce_mean( koDp2 ), voTF.reduce_mean( koDn2 ) )

   def MLossCrossEntropy( aorSelf ) :
      koLabels = aorSelf.voYo
      koLoss = voTF.reduce_mean( voTF.nn.softmax_cross_entropy_with_logits( logits = aorSelf.voOutput, labels = koLabels ) )
      return( koLoss )

   def MOptimizerTriplet( aorSelf ) :
      kdLR = 0.1
      kdRng = 0
      voTF.set_random_seed( kdRng )
      koOptimizer = voTF.train.GradientDescentOptimizer( kdLR ).minimize( aorSelf.vdLossTriplet )
      return( koOptimizer )

   def MOptimizerCrossE( aorSelf ) :
      kdLR = 0.01
      kdRng = 0
      voTF.set_random_seed( kdRng )
      koOptimizer = voTF.train.AdamOptimizer( kdLR ).minimize( aorSelf.vdLossCrossE )
      return( koOptimizer )

   def MTrainModel( aorSelf, aoX, aoY, aiEpochs, aiBatchSize = 100 ) :
      koX, koY = voShuffle( aoX, aoY, random_state = 0 )
      for kiEpoch in range( aiEpochs ) :
         kiI = kiEpoch * aiBatchSize
         if( ( kiI + aiBatchSize ) > aoX.shape[ 0 ] ) :
            kiI = 0
         kdLoss = 0.0
         koInRef = koX[ kiI : kiI + aiBatchSize ]
         koInPos, koInNeg = aorSelf.MGetTriplets( koX, koY, kiI, aiBatchSize )
         _, kdL = aorSelf.voSession.run( [ aorSelf.voOptimizerTriplet, aorSelf.vdLossTriplet ],
                                          feed_dict = { aorSelf.voInRef: koInRef,
                                                         aorSelf.voInPos: koInPos,
                                                         aorSelf.voInNeg: koInNeg } )
         kdLoss = kdLoss + kdL
         aorSelf.MSaveModel( )
         print( 'Loss: %.3f' % ( kdLoss ) )
      koWriter = voTF.summary.FileWriter( "SummaryTrainingModel", aorSelf.voSession.graph )         
      koWriter.close()   

   def MTrainClassifier( aorSelf, aoData, aiEpochs, aiBatchSize = 100 ) :
      for kiEpoch in range( aiEpochs ) :
         koX, koY = aoData.next_batch( aiBatchSize )
         koYc = to_categorical( koY ) # Convert labels to one hot
         koLabels = voNP.zeros( aiBatchSize )
         _, kdLoss = aorSelf.voSession.run( [ aorSelf.voOptimizerCrossE, aorSelf.vdLossCrossE ],
                                            feed_dict = { aorSelf.voInRef: koX,
                                                          aorSelf.voInPos: koX,
                                                          aorSelf.voInNeg: koX,
                                                          aorSelf.voYo: koYc,
                                                          aorSelf.voY: koLabels } )
         if( kiEpoch % 10 == 0 ) :
            print( 'Epoch %d: Train Loss: %.3f' % ( kiEpoch, kdLoss ) )
      aorSelf.MSaveModel( )
      koWriter = voTF.summary.FileWriter( "SummaryTrainingClassifier", aorSelf.voSession.graph )         
      koWriter.close()   

   def MTestModel( aorSelf, aoX ) :
      koY = aorSelf.voSession.run( aorSelf.voOutRef, feed_dict = { aorSelf.voInRef: aoX } )         
      return koY 

   def MGetTriplets( aorSelf, aoX, aoY, aiI, aiSize ) :
      # Create Reference, Positive, and Negative arrays
      koPos = voNP.zeros( shape = ( aiSize, aoX.shape[ 1 ] ), dtype = 'float32' )
      koNeg = voNP.zeros( shape = ( aiSize, aoX.shape[ 1 ] ), dtype = 'float32' )
      for kiB in range( aiSize ) :
         kiIp = voNP.where( aoY == aoY[ aiI + kiB ] )[ 0 ]  # Get indeces of positive labels
         kiIn = voNP.where( aoY != aoY[ aiI + kiB ] )[ 0 ]  # Get indeces of negative labels

         # Randomize indeces
         voNP.random.shuffle( kiIp )
         voNP.random.shuffle( kiIn )

         # Pick samples
         if( kiIp[ 0 ] != aiI + kiB ) :
            koPos[ kiB ] = aoX[ kiIp[ 0 ] ]
         else :
            koPos[ kiB ] = aoX[ kiIp[ 1 ] ]
         koNeg[ kiB ] = aoX[ kiIn[ 0 ] ]

      return( koPos, koNeg )

   def MComputeAccuracy( aorSelf, aoX, aoY ):         
      koLabels = voNP.zeros( 100 )         
      koYOneHot = voNP.zeros( ( 100, 10 ) )         
      koA = aorSelf.voSession.run( aorSelf.voOutput, feed_dict = { aorSelf.voInRef: aoX,
                                                                   aorSelf.voInPos: aoX,
                                                                   aorSelf.voInNeg: aoX,
                                                                   aorSelf.voYo: koYOneHot,
                                                                   aorSelf.voY: koLabels } )         
      kdAccuracyCount = 0         
      koTestY = to_categorical( aoY )  
      
      # one hot labels         
      for kiI in range( koTestY.shape[ 0 ] ) :             
         # determine index of maximum output value             
         kiMax = koA[ kiI ].argmax( axis = 0 )                               
         if( koTestY[ kiI, kiMax ] == 1 ):                 
            kdAccuracyCount = kdAccuracyCount + 1                   
      print( "Accuracy count = " + str( kdAccuracyCount / koTestY.shape[ 0 ] * 100 ) + '%' ) 


   def MSaveModel( aorSelf ):         
      koDir = "./TrainedModels/"         
      koName = "TripletEE"         
      if not os.path.exists( koDir ):             
         os.makedirs( koDir )         
      # Save the latest trained models         
      try:
         aorSelf.voSaver.save( aorSelf.voSession, koDir + koName )  
      except:
         print( 'ERROR: Failed to save Model Parameters' )
 
   def MLoadModel( aorSelf ):         
      # restore the trained model         
      koDir = "./TrainedModels/"         
      koName = "TripletEE"            
      try:
         aorSelf.voSaver.restore( aorSelf.voSession, koDir + koName )
      except:
         print( 'NOTE: Model not trained.' )

