from SimpleGraph import SimpleGraph
from QuadraticGraph import QuadraticGraph
from NNGraph1 import NNGraph1
from NNGraph2 import NNGraph2
from Siamese import Siamese
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def visualize(embed, labels):     
   labelset = set(labels.tolist())    
   fig = plt.figure(figsize=(8,8))     
   ax = fig.add_subplot(111)     
   #fig, ax = plt.subplots()     
   for label in labelset:         
      indices = np.where(labels == label)         
      ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)     
   ax.legend()     
   #fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')     
   plt.show()     
   plt.close() 

def MTestGraphAPI( ) :
   # koSim = SimpleGraph( )
   # koSim.simpleComputation( )

   # koQuad = QuadraticGraph( )
   # koQuad.computeRoot( )

   #koNN1 = NNGraph1( )
   #koNN1.trainAndTestArchitecture( )

   #koNN2 = NNGraph2( )
   #koNN2.trainAndTestArchitectures( )

    # Load MNIST dataset     
    mnist = input_data.read_data_sets('MNIST_data', one_hot = False)     
    mnist_test_labels = mnist.test.labels     
    #mnist_test_onehotlabels = to_categorical(mnist_test_labels) 
    ## for onehot outputs 
 
    siamese = Siamese()     
    siamese.trainSiamese(mnist,1000,128)  # 5000, 128  produces good results     
    #siamese.saveModel()     
    ##siamese.loadModel() 
 
    siamese.trainSiameseForClassification(mnist,1000,128)         
    # Test model     
    embed = siamese.test_model(input = mnist.test.images)     
    embed = embed.reshape([-1, 2]) 
 
    visualize(embed, mnist_test_labels) 
 
    siamese.computeAccuracy(mnist)
