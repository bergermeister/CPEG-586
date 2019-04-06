import os
import sys
import math
import cv2
import numpy as voNP
import matplotlib.pyplot as voPlot
from TePool import TePool
from TeActivation import TeActivation
from TcLayer import TcLayer
from TcLayerC import TcLayerC


def main( ) :

   # Settings for Deep Convolutional Neural Network (Accuracy should be ~92%)
   kiBatchSize = 5
   kiCountFML1 = 6   # Feature Maps in first layer
   kiCountFML2 = 12  # Feature Maps in second layer

   # Create an empty list of CNN Layers
   koCNNLayers = [ ]
   koCNNLayers.append( TcLayerC( ( kiCountFML1, 1, 28, 5 ), TePool.XeAvg, TeActivation.XeRELU ) )
   koCNNLayers.append( TcLayerC( ( kiCountFML2, kiCountFML1, 12, 5 ), TePool.XeAvg, TeActivation.XeRELU ) )

   koNNLayers = [ ]
   koNNLayers.append( TcLayer( ( 50, 4 * 4 * kiCountFML2 ), TeActivation.XeRELU, False, 0.8, 0.8 ) )

if __name__ == "__main__" :
    main( )
