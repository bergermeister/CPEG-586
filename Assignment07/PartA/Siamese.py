import tensorflow as tf 
import os 
from tensorflow.examples.tutorials.mnist import input_data 
from keras.utils import to_categorical 
import numpy as np 
 
class Siamese(object): 
   def __init__(self):        
      #----set up place holders for inputs and labels for the siamese network---         
      # two input placeholders for Siamese network         
      self.tf_inputA = tf.placeholder(tf.float32, [None, 784], name = 'inputA')         
      self.tf_inputB = tf.placeholder(tf.float32, [None, 784], name = 'inputB')         
     
      # labels for the image pair        # 1: similar, 0: dissimilar         
      self.tf_Y = tf.placeholder(tf.float32, [None,], name = 'Y')         
      self.tf_YOneHot = tf.placeholder(tf.float32, [None,10], name = 'YoneHot')         
     
      # outputs, loss function and training optimizer         
      self.outputA, self.outputB = self.siameseNetwork()         
      self.output = self.siameseNetworkWithClassification()         
      self.loss = self.contrastiveLoss()         
      self.lossCrossEntropy = self.crossEntropyLoss() 
      self.optimizer = self.optimizer_initializer()         
      self.optimizerCrossEntropy = self.optimizer_initializer_crossEntropy()         
      self.saver = tf.train.Saver()                  
      
      # Initialize tensorflow session         
      self.sess = tf.Session()         
      self.sess.run(tf.global_variables_initializer()) 
 
   def layer(self, tf_input, num_hidden_units, variable_name, trainable=True):         
      # tf_input: batch_size x n_features         
      # num_hidden_units: number of hidden units         
      tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)         
      num_features = tf_input.get_shape()[1]         
      W = tf.get_variable( name = variable_name + '_W',              
                           dtype = tf.float32,              
                           shape = [num_features, num_hidden_units],              
                           initializer = tf_weight_initializer,             
                           trainable=trainable )
      b = tf.get_variable( name = variable_name + '_b',              
                           dtype = tf.float32,
                           shape = [num_hidden_units], 
                           trainable=trainable ) 
      out = tf.add(tf.matmul(tf_input, W), b) 
      
      return out 
 
   def network(self, tf_input, trainable=True):         
      # Setup FNN         
      fc1 = self.layer(tf_input = tf_input, num_hidden_units = 512, trainable=trainable, variable_name = 'fc1')         
      ac1 = tf.nn.relu(fc1)         
      fc2 = self.layer(tf_input = ac1, num_hidden_units = 512, trainable=trainable, variable_name = 'fc2')         
      ac2 = tf.nn.relu(fc2)         
      fc3 = self.layer(tf_input = ac2, num_hidden_units = 100, trainable=trainable, variable_name = 'fc3')         
      return fc3 
 
   def networkWithClassification(self, tf_input):         
      # Setup FNN         
      fc3 = self.network(tf_input, trainable=False)         
      ac3 = tf.nn.relu(fc3)         
      fc4 = self.layer(tf_input = ac3, num_hidden_units = 80, trainable=True, variable_name = 'fc4')         
      ac4 = tf.nn.relu(fc4)         
      fc5 = self.layer(tf_input = ac4, num_hidden_units = 10, trainable=True, variable_name = 'fc5')          
      return fc5 
 
   def siameseNetwork(self):        
      # Initialze neural network         
      with tf.variable_scope("siamese") as scope: 
         outputA = self.network(self.tf_inputA)             
         # share weights             
         scope.reuse_variables()             
         outputB = self.network(self.tf_inputB)         
      return outputA, outputB 
 
   def siameseNetworkWithClassification(self):         
      # Initialze neural network         
      with tf.variable_scope("siamese",reuse=tf.AUTO_REUSE) as scope:         
      #with tf.variable_scope("siamese") as scope:             
         output = self.networkWithClassification(self.tf_inputA)         
      return output 
 
   def contrastiveLoss(self, margin = 5.0):         
      with tf.variable_scope("siamese") as scope:
         labels = self.tf_Y             
         # Euclidean distance squared             
         dist = tf.pow(tf.subtract(self.outputA, self.outputB), 2, name = 'Dw')             
         Dw = tf.reduce_sum(dist, 1)             
         # add a small value 1e-6 to increase the stability of calculating the gradients for sqrt             
         Dw2 = tf.sqrt(Dw + 1e-6, name = 'Dw2')             
         # Loss function             
         lossSimilar = tf.multiply(labels, tf.pow(Dw2,2), name = 'constrastiveLoss_1')             
         lossDissimilar = tf.multiply(tf.subtract(1.0, labels), tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2), name = 'constrastiveLoss_2')             
         loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name = 'constrastiveLoss')         
      return loss 
 
   def crossEntropyLoss(self):         
      labels = self.tf_YOneHot         
      lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=labels))         
      return lossd 
 
   def optimizer_initializer(self):         
      LEARNING_RATE = 0.01         
      RAND_SEED = 0 # random seed         
      tf.set_random_seed(RAND_SEED)         
      # Initialize optimizer         
      optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)         
      return optimizer 
 
   def optimizer_initializer_crossEntropy(self):         
      LEARNING_RATE = 0.01        
      RAND_SEED = 0 # random seed         
      tf.set_random_seed(RAND_SEED)         
      # Initialize optimizer         
      optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.lossCrossEntropy)         
      return optimizer  
 
   def trainSiamese(self, mnist,numIterations, batchSize=100):         
      # Train the network for embeddings via contrastive loss         
      for i in range(numIterations):             
         input1, y1 = mnist.train.next_batch(batchSize)            
         input2, y2 = mnist.train.next_batch(batchSize)             
         label = (y1 == y2).astype('float')             
         _, trainingLoss = self.sess.run([self.optimizer, self.loss],                  
                                         feed_dict = {self.tf_inputA: input1, self.tf_inputB: input2, self.tf_Y: label})             
         if i % 50 == 0:                 
            print('iteration %d: train loss %.3f' % (i, trainingLoss)) 
 
   def trainSiameseForClassification(self, mnist,numIterations, batchSize=10):         
      # Train the network for classification via softmax         
      for i in range(numIterations):             
         input1, y1 = mnist.train.next_batch(batchSize)             
         y1c = to_categorical(y1)  # convert labels to one hot             
         labels = np.zeros(batchSize)             
         _, trainingLoss = self.sess.run([self.optimizerCrossEntropy, self.lossCrossEntropy],                  
                                         feed_dict = {self.tf_inputA: input1, self.tf_inputB: input1,self.tf_YOneHot: y1c, self.tf_Y:labels})             
         if i % 10 == 0:                
            print('iteration %d: train loss %.3f' % (i, trainingLoss)) 
 
   def test_model(self, input):         
      # Test the trained model         
      output = self.sess.run(self.outputA, feed_dict = {self.tf_inputA: input})         
      return output 
 
   def saveModel(self):         
      modelDir = "./TrainedModels/"         
      modelName = "SiameseAM"         
      if not os.path.exists(modelDir):             
         os.makedirs(modelDir)         
         # Save the latest trained models         
      self.saver.save(self.sess, modelDir + modelName)  
 
   def loadModel(self):         
      # restore the trained model         
      modelDir = "d:/deeplearning/trainedmodels/"         
      modelName = "SiameseAM"         
      #assert os.path.exists(modelDir + modelName)         
      self.saver.restore(self.sess, modelDir + modelName)          
      
   def saveWeights(self, w1, b1, w2, b2):         
      weights = {'w1': w1,                    
                 'b1': b1,                    
                 'w2': w2,                    
                 'b2': b2}         
      with open('savedWeights.pickle', 'wb') as handle:             
         pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)         
      print('Weights saved successfully.')  
  
   def computeAccuracy(self,mnist):         
      labels = np.zeros(100)         
      yonehot = np.zeros((100,10))         
      aout = self.sess.run(self.output, feed_dict={self.tf_inputA: mnist.test.images,            
                                                   self.tf_inputB: mnist.test.images,self.tf_YOneHot: yonehot, self.tf_Y:labels})         
      accuracyCount = 0         
      testY = to_categorical(mnist.test.labels)  
      
      # one hot labels         
      for i in range(testY.shape[0]):             
         # determine index of maximum output value             
         maxindex = aout[i].argmax(axis = 0)                               
         if (testY[i,maxindex] == 1):                 
            accuracyCount = accuracyCount + 1                   
      print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%') 