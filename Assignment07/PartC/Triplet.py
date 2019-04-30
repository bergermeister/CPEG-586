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
      aorSelf.vdLossTriplet = aorSelf.MLossTriplet( aorSelf.voOutRef, aorSelf.voOutPos, aorSelf.voOutNeg )
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

      return( koFlat )

   def MNetworkTriplet( aorSelf ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         koOutRef = aorSelf.MCNN( aorSelf.voInRef )
         
         # Share weights
         koScope.reuse_variables( )
         koOutPos = aorSelf.MCNN( aorSelf.voInPos )

         # Share weights
         #koScope.reuse_variables( )
         koOutNeg = aorSelf.MCNN( aorSelf.voInNeg )

      return( koOutRef, koOutPos, koOutNeg )

   def MNetworkClassifier( aorSelf ) :
      # Initialize Neural Network
      with voTF.variable_scope( "triplet", reuse = voTF.AUTO_REUSE ) as koScope :
         koCNN = aorSelf.MCNN( aorSelf.voInRef, abTrainable = False )
         koA1  = voTF.nn.relu( koCNN )

         # Setup Dense Layers
         koLD1 = aorSelf.MLayerD( koA1, [ koA1.get_shape( )[ 1 ], 100 ], TeActivation.XeSoftMax, 'LD1' )
         koLD2 = aorSelf.MLayerD( koLD1,  [ 100, 10 ], TeActivation.XeNone, 'LD2' )

      return( koLD2 )

   def MLossTriplet( aorSelf, aoX, aoXp, aoXn, adMargin = 0.2 ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         koDp2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( aoX, aoXp ) ), 1 )
         koDn2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( aoX, aoXn ) ), 1 )

         koLoss = voTF.maximum( 0.0, voTF.math.add( voTF.math.subtract( koDp2, koDn2 ), adMargin ) )

      return( voTF.reduce_mean( koLoss ) ) #, voTF.reduce_mean( koDp2 ), voTF.reduce_mean( koDn2 ) )

   def MLossCrossEntropy( aorSelf ) :
      koLabels = aorSelf.voYo
      koLoss = voTF.reduce_mean( voTF.nn.softmax_cross_entropy_with_logits( logits = aorSelf.voOutput, labels = koLabels ) )
      return( koLoss )

   def MOptimizerTriplet( aorSelf ) :
      kdLR = 0.01
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
      for kiEpoch in range( aiEpochs ) :
         kdLoss = 0.0
         koInRef, koInPos, koInNeg = aorSelf.MGetTriplets( aoX, aoY, 10000 )
         for kiI in range( 0, int( len( koInRef ) ), aiBatchSize ) :            
            _, kdL = aorSelf.voSession.run( [ aorSelf.voOptimizerTriplet, aorSelf.vdLossTriplet ],
                                            feed_dict = { aorSelf.voInRef: koInRef[ kiI : kiI + aiBatchSize ], 
                                                          aorSelf.voInPos: koInPos[ kiI : kiI + aiBatchSize ], 
                                                          aorSelf.voInNeg: koInNeg[ kiI : kiI + aiBatchSize ] } )
            kdLoss = kdLoss + kdL
         print( 'Loss: %.3f' % ( kdLoss ) )

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

   def MTestModel( aorSelf, aoX ) :
      koY = aorSelf.voSession.run( aorSelf.voOutRef, feed_dict = { aorSelf.voInRef: aoX } )         
      return koY 

   def MGetTriplets( aorSelf, aoX, aoY, aiSize ) :
      # Randomize inputs
      koX, koY = voShuffle( aoX, aoY, random_state = 0 )

      # Create Reference, Positive, and Negative arrays
      koRef = voNP.zeros( shape = ( aiSize, aoX.shape[ 1 ] ), dtype = 'float32' )
      koPos = voNP.zeros( shape = ( aiSize, aoX.shape[ 1 ] ), dtype = 'float32' )
      koNeg = voNP.zeros( shape = ( aiSize, aoX.shape[ 1 ] ), dtype = 'float32' )
      for kiI in range( aiSize ) :
         # Select 2 random labels
         kiSelect = voNP.random.choice( aoY, 2, replace = False )
         koYp = kiSelect[ 0 ]                   # Positive label
         koYn = kiSelect[ 1 ]                   # Negative label
         kiIp = voNP.where( aoY == koYp )[ 0 ]  # Get indeces of positive labels
         kiIn = voNP.where( aoY == koYn )[ 0 ]  # Get indeces of negative labels

         # Randomize indeces
         voNP.random.shuffle( kiIp )
         voNP.random.shuffle( kiIn )

         # Pick samples
         koRef[ kiI ] = aoX[ kiIp[ 0 ] ]
         koPos[ kiI ] = aoX[ kiIp[ 1 ] ]
         koNeg[ kiI ] = aoX[ kiIn[ 0 ] ]

      return( koRef, koPos, koNeg )

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
      aorSelf.voSaver.save( aorSelf.voSession, koDir + koName )  
 
   def MLoadModel( aorSelf ):         
      # restore the trained model         
      koDir = "./TrainedModels/"         
      koName = "TripletEE"            
      try:
         aorSelf.voSaver.restore( aorSelf.voSession, koDir + koName )
      except:
         print( 'Model not trained.' )

