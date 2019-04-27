import tensorflow as voTF
import numpy as voNP
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
      aorSelf.voOptimzier = aorSelf.MInitOptimizer( )
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
      koLC2 = aorSelf.MLayerC( koLC1, [ 32,   64 ], [ 3, 3 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC2' )
      #koLC3 = aorSelf.MLayerC( koLC2, [ 64,  128 ], [ 3, 3 ], [ 2, 2 ], TeActivation.XeRELU, TePool.XeMax, 'LC3' )

      # Flatten the output
      koFlat = voTF.reshape( koLC2, [ -1, 7 * 7 * 64 ] )

      # Setup Dense Layers
      koLD1 = aorSelf.MLayerD( koFlat, [ koFlat.get_shape( )[ 1 ], 100 ], TeActivation.XeRELU, 'LD1' )
      koLD2 = aorSelf.MLayerD( koLD1,  [ 100, 10 ], TeActivation.XeSoftMax, 'LD2' )

      return( koLD2 )

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

   def MLoss( aorSelf, aoX, aoXp, aoXn, adMargin = 5.0 ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         # L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
         koDp2 = voTF.square( aorSelf.MEuclideanDistance( aoX, aoXp ) )
         koDn2 = voTF.square( aorSelf.MEuclideanDistance( aoX, aoXn ) )
         
         koLoss = voTF.maximum( 0.0, voTF.add( voTF.subtract( koDp2, koDn2 ), adMargin ) )

      return( voTF.reduce_mean( koLoss ), voTF.reduce_mean( koDp2 ), voTF.reduce_mean( koDn2 ) )
 
   def MEuclideanDistance( aorSelf, aoX, aoY ) :
      koD = voTF.square( voTF.subtract( aoX, aoY ) )
      koD = voTF.sqrt( voTF.reduce_sum( koD ) )
      return( koD )

   def MInitOptimizer( aorSelf ) :
      kdLR = 0.01
      kdRng = 0
      voTF.set_random_seed( kdRng )
      koOptimizer = voTF.train.GradientDescentOptimizer( kdLR ).minimize( aorSelf.vdLoss[ 0 ] )
      return( koOptimizer )

   def MTrain( aorSelf, aoMNIST, aoIterations, aiBatchSize = 100 ) :
      # Train the network for embeddings
      for kiI in range( aoIterations ) :
         koTriplet = aorSelf.MGetTriplet( aoMNIST )
         koBatch, kiY = aoMNIST.train.next_batch( aiBatchSize )
         
         _, kdLoss = aorSelf.voSession.run( [ aorSelf.voOptimizer, aorSelf.vdLoss ],
                                            feed_dict = { aorSelf.voInRef: koInput1, 
                                                          aorSelf.voInPos: koInput2, 
                                                          aorSelf.voInNeg: koInput3 } )

   #def MGetBatch( aorSelf, n_labels, n_triplets=1, is_target_set_train=True ) :
   
   def MGetTriplet( aorSelf, aoData ) :
      # Get a pair of choices
      koI = voNP.random.choice( aoData.labels, 2, replace = False )
      koOutPos = koI[ 0 ]      # Choice 0 is the positive image
      koOutNeg = koI[ 1 ]      # Choice 1 is the negative image

      # Get all indeces of positive imnages
      koI = voNP.where( aoY == koOutPos )[ 0 ]
      voNP.random.shuffle( koI )
      
      # Select a pair of images from the same label
      koInRef = aoX[ koI[ 0 ], :, :, : ]
      koInPos = aoX[ koI[ 1 ], :, :, : ]

      # Select an image from a different label
      koI = voNP.where( koY == koOutNeg )[ 0 ]
      voNP.random.shuffle( koI )
      koInNeg = aoX[ koI[ 0 ], :, :, : ]
           
      return koInRef, koInPos, koInNeg, koOutPos, koOutPos, koOutNeg



  #     if is_target_set_train:
  #
  #         target_data = self.train_data
  #
  #         target_labels = self.train_labels
  #
  #     else:
  #
  #         target_data = self.validation_data
  #
  #         target_labels = self.validation_labels
  #
  #
  #
  #     c = target_data.shape[3]
  #
  #     w = target_data.shape[1]
  #
  #     h = target_data.shape[2]
  #
  #
  #
  #     data_a = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
  #
  #     data_p = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
  #
  #     data_n = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
  #
  #     labels_a = numpy.zeros(shape=n_triplets, dtype='float32')
  #
  #     labels_p = numpy.zeros(shape=n_triplets, dtype='float32')
  #
  #     labels_n = numpy.zeros(shape=n_triplets, dtype='float32')
  #
  #
  #
  #     for i in range(n_triplets):
  #
  #         data_a[i, :, :, :], data_p[i, :, :, :], data_n[i, :, :, :], \
  #
  #         labels_a[i], labels_p[i], labels_n[i] = \
  #
  #             get_one_triplet(target_data, target_labels)
  #
  #
  #
  #     return data_a, data_p, data_n, labels_a, labels_p, labels_n