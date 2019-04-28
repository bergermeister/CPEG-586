import tensorflow as voTF
import numpy as voNP
from sklearn.utils import shuffle as voShuffle
from Activation import TeActivation
from Pool import TePool

class TcTriplet( object ) :
   def __init__( aorSelf ) :
      # Set up place holders for inputs
      aorSelf.voInRef = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InRef' )  # Ref image input
      aorSelf.voInPos = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InPos' ) # Same image input
      aorSelf.voInNeg = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'InNeg' ) # Diff image input

      # Set up place holders for labels
      aorSelf.voY = voTF.placeholder( voTF.float32, [ None, 10 ], name = 'voY' )

      # Setup the Outputs, Loss Function, and Training Optimizer
      aorSelf.voOutRef, aorSelf.voOutPos, aorSelf.voOutNeg = aorSelf.MSetupTriplet( )
      aorSelf.vdLoss = aorSelf.MLoss( aorSelf.voOutRef, aorSelf.voOutPos, aorSelf.voOutNeg )
      aorSelf.voOptimizer = aorSelf.MInitOptimizer( )
      aorSelf.voSaver = voTF.train.Saver( )

      # Initialize TensorFlow session
      aorSelf.voSession = voTF.Session( )
      aorSelf.voSession.run( voTF.global_variables_initializer( ) )

   def MSetupTriplet( aorSelf ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         koOutRef = aorSelf.MSetupNetwork( aorSelf.voInRef )
         
         # Share weights
         koScope.reuse_variables( )
         koOutPos = aorSelf.MSetupNetwork( aorSelf.voInPos )

         # Share weights
         koScope.reuse_variables( )
         koOutNeg = aorSelf.MSetupNetwork( aorSelf.voInNeg )

      return( koOutRef, koOutPos, koOutNeg )

   def MSetupNetwork( aorSelf, aoInput, abTrainable = True ) :
      # Reshape the input
      koX = voTF.reshape( aoInput, [ -1, 28, 28, 1 ] )

      # Setup Convolutional Layers
      koLC1 = aorSelf.MLayerC( koX,   [  1,   32 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC1' )
      koLC2 = aorSelf.MLayerC( koLC1, [ 32,   64 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC2' )
      koLC3 = aorSelf.MLayerC( koLC2, [ 64,  128 ], [ 5, 5 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC3' )

      # Flatten the output
      #koFlat = voTF.reshape( koLC2, [ -1, 7 * 7 * 64 ] )
      koFlat = voTF.reshape( koLC3, [ -1, 4 * 4 * 128 ] )

      # Setup Dense Layers
      koLD1 = aorSelf.MLayerD( koFlat, [ koFlat.get_shape( )[ 1 ], 10 ], TeActivation.XeSoftMax, 'LD1' )
      #koLD2 = aorSelf.MLayerD( koLD1,  [ 100, 10 ], TeActivation.XeSoftMax, 'LD2' )

      return( koLD1 )

   def MLayerC( aorSelf, aoX, aoShapeLayer, aoShapeKernel, aoShapePool, aeActivation, aePool, aoVarName ) :
      # Setup Shapes: Kernel Rows           Kernel Cols         Input              Num Kernels
      koShapeKernel = [ aoShapeKernel[ 0 ], aoShapeKernel[ 1 ], aoShapeLayer[ 0 ], aoShapeLayer[ 1 ] ]

      # Initialise weights and bias for the kernel
      koW = voTF.get_variable( name = aoVarName + '_W',
                              dtype = voTF.float32,
                              shape = koShapeKernel,
                              initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                              trainable = True )
      koB = voTF.get_variable( name = aoVarName + '_B',
                              dtype = voTF.float32,
                              shape = aoShapeLayer[ 1 ],
                              initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                              trainable = True )

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

   def MLayerD( aorSelf, aoX, aoShape, aeActivation, aoVarName ) :
      # Initialize weights and bias
      koW = voTF.get_variable( name = aoVarName + '_W', 
                               dtype = voTF.float32, 
                               shape = aoShape,
                               initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                               trainable = True )
      koB = voTF.get_variable( name = aoVarName + '_B', 
                               dtype = voTF.float32, 
                               shape = [ aoShape[ 1 ] ],
                               initializer = voTF.random_normal_initializer( mean = 0, stddev = 0.01 ),
                               trainable = True )

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

   def MLoss( aorSelf, aoX, aoXp, aoXn, adMargin = 0.1 ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         koDp2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( aoX, aoXp ) ), 1 )
         koDn2 = voTF.reduce_sum( voTF.square( voTF.math.subtract( aoX, aoXn ) ), 1 )

         koLoss = voTF.maximum( 0.0, voTF.math.add( voTF.math.subtract( koDp2, koDn2 ), adMargin ) )

      return( voTF.reduce_mean( koLoss ), voTF.reduce_mean( koDp2 ), voTF.reduce_mean( koDn2 ) )
         #koNp = aorSelf.MEuclideanDistance( aoX, aoXp )
         #koNn = aorSelf.MEuclideanDistance( aoX, aoXn )
         #
         #koEp = voTF.math.exp( koNp )
         #koEn = voTF.math.exp( koNn )
         #
         #koDp = voTF.math.divide( koEp, voTF.math.add( koEp, koEn ) )
         #koDn = voTF.math.divide( koEn, voTF.math.add( koEp, koEn ) )
         #
         #koLoss = voTF.math.divide( koDp, koDn )

         # koDp2 = voTF.square( aorSelf.MEuclideanDistance( aoX, aoXp ) )
         # koDn2 = voTF.square( aorSelf.MEuclideanDistance( aoX, aoXn ) )
         
         # koLoss = voTF.math.divide( koDp2, voTF.subtract( koDn2, adMargin ) )
         # voTF.maximum( 0.0, voTF.add( voTF.subtract( koDp2, koDn2 ), adMargin ) )
         

      #return( voTF.reduce_mean( koLoss ), voTF.reduce_mean( koDp ), voTF.reduce_mean( koDn ) )
      #return( voTF.reduce_mean( koLoss ), voTF.reduce_mean( koDp2 ), voTF.reduce_mean( koDn2 ) )
 
   def MEuclideanDistance( aorSelf, aoX, aoY ) :
      koD = voTF.square( voTF.subtract( aoX, aoY ) )
      koD = voTF.sqrt( voTF.reduce_sum( koD ) )
      return( koD )

   def MInitOptimizer( aorSelf ) :
      kdLR = 0.1
      kdRng = 0
      voTF.set_random_seed( kdRng )
      koOptimizer = voTF.train.GradientDescentOptimizer( kdLR ).minimize( aorSelf.vdLoss[ 0 ] )
      return( koOptimizer )

   def MTrain( aorSelf, aoX, aoY, aiEpochs, aiBatchSize = 100 ) :
      for kiEpoch in range( aiEpochs ) :
         koInRef, koInPos, koInNeg = aorSelf.MGetTriplets( aoX, aoY, 10000 )
         for kiI in range( 0, int( len( koInRef ) ), aiBatchSize ) :            
            _, kdLoss = aorSelf.voSession.run( [ aorSelf.voOptimizer, aorSelf.vdLoss ],
                                               feed_dict = { aorSelf.voInRef: koInRef[ kiI : kiI + aiBatchSize ], 
                                                             aorSelf.voInPos: koInPos[ kiI : kiI + aiBatchSize ], 
                                                             aorSelf.voInNeg: koInNeg[ kiI : kiI + aiBatchSize ] } )
         print( 'Loss: %.3f' % ( kdLoss[ 0 ] ) )

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

         ## Store the current element as the reference / anchor
         #koRef[ kiI ] = aoX[ kiI ]
         #
         ## Get the label of the current element
         #koLabel = aoY[ kiI ]
         #
         ## Get list of indeces for positives and negatives
         #kiIp = voNP.where( aoY == koLabel )[ 0 ]
         #kiIn = voNP.where( aoY != koLabel )[ 0 ]
         #
         ## Randomize the choice
         #voNP.random.shuffle( kiIp )
         #voNP.random.shuffle( kiIn )
         #
         #if( kiIp[ 0 ] != kiI ) :
         #   koPos[ kiI ] = aoX[ kiIp[ 0 ] ]
         #else :
         #   koPos[ kiI ] = aoX[ kiIp[ 1 ] ]
         #
         #koNeg[ kiI ] = aoX[ kiIn[ 0 ] ]

      return( koRef, koPos, koNeg )

   def MTest( aorSelf, aoX ) :
      koY = aorSelf.voSession.run( aorSelf.voOutRef, feed_dict = { aorSelf.voInRef: aoX } )         
      return koY 