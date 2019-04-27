import sys
from Triplet import TcTriplet
from tensorflow.examples.tutorials.mnist import input_data

def main( ) :
   # Load the MNIST dataset
   koMNIST = input_data.read_data_sets( 'MNIST_data', one_hot = False )
   koLabels = koMNIST.test.labels

   koTriplet = TcTriplet( )
   koTriplet.MTrain( koMNIST.train, 1000, 128 )

if __name__ == "__main__":     
   main( )