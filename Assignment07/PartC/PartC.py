import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Triplet import TcTriplet
from tensorflow.examples.tutorials.mnist import input_data

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

def main( ) :
   # Load the MNIST dataset
   koMNIST = input_data.read_data_sets( 'MNIST_data', one_hot = False )
   koLabels = koMNIST.test.labels

   koTriplet = TcTriplet( )
   koTriplet.MTrain( koMNIST.train.images, koMNIST.train.labels, 30, 250 )

   koOut = koTriplet.MTest( koMNIST.test.images )
   koOut = koOut.reshape( [ -1, 2 ] ) 

   visualize( koOut, koLabels )

if __name__ == "__main__":     
   main( )