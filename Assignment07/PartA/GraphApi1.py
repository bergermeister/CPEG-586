from SimpleGraph import SimpleGraph
from QuadraticGraph import QuadraticGraph
from NNGraph1 import NNGraph1
from NNGraph2 import NNGraph2
from Siamese import Siamese
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def MVisualize( aoEmbed, aoLabels ) :
   koLabelSet = set( aoLabels.toList( ) )
   koFig = plt.figure( figsize=( 8, 8 ) )
   koAX = koFig.add_subplot( 111 )

   for koLabel in koLabelSet :
      koIndices = np.where( aoLabels == koLabel )
      koAX.scatter( aoEmbed[ koIndices, 0 ], embded[ indices, 1 ], label = label, s = 20 )
   koAX.legend( )
   plt.show( )
   plt.close( )

def MTestGraphAPI( ) :
   # koSim = SimpleGraph( )
   # koSim.simpleComputation( )

   # koQuad = QuadraticGraph( )
   # koQuad.computeRoot( )

   #koNN1 = NNGraph1( )
   #koNN1.trainAndTestArchitecture( )

   #koNN2 = NNGraph2( )
   #koNN2.trainAndTestArchitectures( )

   koMNIST = input_data.read_data_sets( 'MNIST_data', one_hot = False )
   koMNISTLabels = koMNIST.test.labels

   koSiamese = Siamese( )
   koSiamese.trainSiamese( koMNIST, 5000, 100 )

   # Test Model
   koEmbed = koSiamese.test_model( input = koMNIST.test.images )
   koEmbed = koEmbed.reshape( [ -1, 2 ] )
   MVisualize( koEmbed, koMNISTLabels )
