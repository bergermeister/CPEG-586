import tensorflow as voTF

class TcTriplet( object ) :
   def __init__( aorSelf ) :
      # Set up place holders for inputs
      aorSelf.voInRef = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'voInRef' ) # Ref image input
      aorSelf.voInPos = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'voInPos' ) # Same image input
      aorSelf.voInNeg = voTF.placeholder( voTF.float32, [ None, 784 ], name = 'voInNeg' ) # Diff image input

      # Set up place holders for labels
      aorSelf.voY = voTF.placeholder( voTF.float32, [ None, ], name = 'voY' )
      # TODO

      # Setup the Outputs, Loss Function, and Training Optimizer
      aorSelf.voOutRef, aorSelf.voOutPos, aorSelf.voOutNeg = aorSelf.MSetupTriplet( )
      # TODO aorSelf.voOutput = aorSelf.voMSetupClassification
      aorSelf.vdLoss = aorSelf.MLoss( )
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
      # Setup FNN
      koFC1 = aorSelf.MLayer( aoInput = aoInput, aoNumHidden = 512, abTrainable = abTrainable, aoVarName = 'koFC1' )
      koAC1 = voTF.nn.relu( koFC1 )
      koFC2 = aorSelf.MLayer( aoInput = koAC1, aoNumHidden = 512, abTrainable = abTrainable, aoVarName = 'koFC2' )
      koAC2 = voTF.nn.relu( koFC2 )
      koFC3 = aorSelf.MLayer( aoInput = koAC2, aoNumHidden = 100, abTrainable = abTrainable, aoVarName = 'koFC3' )

      return( koFC3 )

   def MLayer( aorSelf, aoInput, aoNumHidden, aoVarName, abTrainable = True ) :
      koTFwInit = voTF.random_normal_initializer( mean = 0, stddev = 0.01 )
      koNumFeat = aoInput.get_shape( )[ 1 ]
      koW = voTF.get_variable( name = aoVarName + '_W', 
                               dtype = voTF.float32, 
                               shape = [ koNumFeat, aoNumHidden ],
                               initializer = koTFwInit,
                               trainable = abTrainable )
      koB = voTF.get_variable( name = aoVarName + '_B', 
                               dtype = voTF.float32, 
                               shape = [ aoNumHidden ],
                               initializer = koTFwInit,
                               trainable = abTrainable )

      koOut = voTF.add( voTF.matmul( aoInput, koW ), koB )

      return( koOut )

   def MLoss( aorSelf, adMargin = 5.0 ) :
      with voTF.variable_scope( "triplet" ) as koScope :
         kdLoss = 0.0

      return( kdLoss )

   def MInitOptimizer( aorSelf ) :
      kdLR = 0.01
      kdRng = 0
      voTF.set_random_seed( kdRng )
      koOptimizer = voTF.train.GradientDescentOptimizer( kdLR ).minimize( aorSelf.vdLoss )
      return( koOptimizer )

   def MTrain( aorSelf, aoMNIST, aoIterations, aiBatchSize = 100 ) :
      # Train the network for embeddings
      for kiI in range( aoIterations ) :
         koInput1, kiY1 = aoMNIST.train.next_batch( aiBatchSize )
         koInput2, kiY2 = aoMNIST.train.next_batch( aiBatchSize )
         koInput3, kiY3 = aoMNIST.train.next_batch( aiBatchSize )
         _, kdLoss = aorSelf.voSession.run( [ aorSelf.voOptimizer, aorSelf.vdLoss ],
                                            feed_dict = { aorSelf.voInRef: koInput1, 
                                                          aorSelf.voInPos: koInput2, 
                                                          aorSelf.voInNeg: koInput3 } )
