import numpy as np   
import pandas as pd   
import tensorflow as tf   
import matplotlib.pyplot as plt 
 
class NNGraph2(object):     
   def __init__(self):         
      self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.readTrainTestData()         
      # model parameters for three architectures         
      self.num_hidden_nodes = [5, 10, 20]   
 
      self.lossPlotData = {5: [], 10: [], 20: []}           
      self.W1 = {5: None, 10: None, 20: None}           
      self.W2 = {5: None, 10: None, 20: None}         
      self.b1 = {5: None, 10: None, 20: None}           
      self.b2 = {5: None, 10: None, 20: None}           
      self.numEpochs = 500         
      for hidden_node in self.num_hidden_nodes:              
         # initialize weight matrices and biases for the three architectures             
         self.W1[hidden_node] = np.random.rand(4, hidden_node)             
         self.W2[hidden_node] = np.random.rand(hidden_node, 3)             
         self.b1[hidden_node] = np.random.rand(hidden_node)             
         self.b2[hidden_node] = np.random.rand(3) 
 
   def readTrainTestData(self):         
      # Download dataset         
      IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"           
      IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv" 
 
      names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']           
      train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)           
      test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1) 
 
      # drop the species column - now Xtrain = 120x4, Xtest=30x4         
      Xtrain = train.drop("species", axis=1)           
      Xtest = test.drop("species", axis=1) 
 
      # Encode class values into one-hot e.g., class 2 = 1 0 0         
      ytrain = pd.get_dummies(train.species)           
      ytest = pd.get_dummies(test.species)         
      return Xtrain, ytrain, Xtest, ytest 
 
   def plotLossComparison(self):         
      plt.figure(figsize=(12,8))         
      for hidden_node in self.num_hidden_nodes:               
         plt.plot(range(self.numEpochs), self.lossPlotData[hidden_node], label="nn: 4-%d-3" % hidden_node)        
      plt.xlabel('Iteration', fontsize=12)           
      plt.ylabel('Loss', fontsize=12)          
      plt.legend(fontsize=12)           
      plt.show() 
 
   def trainAndTestArchitectures(self):         
      # do training for 3 different network architectures-(4-5-3) (4-10-3) (4-20-3)         
      # Important: Each model's training is run as a separate tensorflow session         
      for hidden_node in self.num_hidden_nodes:               
         self.trainNNmodel(hidden_node)
         
      # session is closed so plotting is done outside of tensorflow's session         
      self.plotLossComparison() 
 
      # another tensorflow session is created to compute accuracies on test data         
      self.computeAccuracy() 
 
   def computeAccuracy(self):         
      # Evaluate models on the test set         
      X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')          
      y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y') 
 
      for hidden_node in self.num_hidden_nodes:             
         # evaluate the network based on trained weights             
         W1 = tf.Variable(self.W1[hidden_node])             
         W2 = tf.Variable(self.W2[hidden_node])             
         b1 = tf.Variable(self.b1[hidden_node])             
         b2 = tf.Variable(self.b2[hidden_node])             
         A1 = tf.sigmoid(tf.add(tf.matmul(X, W1),b1))             
         A2 = tf.sigmoid(tf.add(tf.matmul(A1, W2),b2)) 
 
         # Calculate the predicted outputs             
         init = tf.global_variables_initializer()            
         with tf.Session() as sess:                 
            sess.run(init)                 
            A2p = sess.run(A2, feed_dict={X: self.Xtest, y: self.ytest})             
         sess.close()    
 
         # Calculate the accuracy             
         correct = [estimate.argmax(axis=0) == target.argmax(axis=0)                         
                    for estimate, target in zip(A2p, self.ytest.values)]             
         accuracy = 100 * sum(correct) / len(correct)             
         print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_node, accuracy)) 
 
   def trainNNmodel(self, hidden_node):         
      tf.reset_default_graph() # reset the graph as this will be called multiple times                                  
      # for different models (numer of hidden nodes)         
      # placeholders for input and expected outputs         
      batchSize = 10         
      X = tf.placeholder(shape=(batchSize, 4), dtype=tf.float64, name='X')         
      y = tf.placeholder(shape=(batchSize, 3), dtype=tf.float64, name='y') 
 
      # tf variables for weight matrices and biases         
      W1 = tf.Variable(self.W1[hidden_node], dtype=tf.float64)         
      W2 = tf.Variable(self.W2[hidden_node], dtype=tf.float64)         
      b1 = tf.Variable(self.b1[hidden_node], dtype=tf.float64)         
      b2 = tf.Variable(self.b2[hidden_node], dtype=tf.float64)         
      # neural net graph         
      A1 = tf.sigmoid(tf.add(tf.matmul(X, W1), b1))         
      A2 = tf.sigmoid(tf.add(tf.matmul(A1, W2), b2))         
      #A2 = tf.nn.softmax(tf.matmul(A1, W2) + b2)  # use with cross entropy loss 
 
      # define loss          
      error = tf.square(A2 - y)         
      loss = tf.reduce_sum(error)         
      #cross_entropy_error = -tf.reduce_sum(y*tf.log(A2 + 1e-10))         
      #loss = tf.reduce_sum(cross_entropy_error) 
 
      # define train function to minimize the loss         
      #optimizer = tf.train.GradientDescentOptimizer(0.05) 
      optimizer = tf.train.AdamOptimizer(learning_rate=0.1) # the optimizer         
      train = optimizer.minimize(loss) 
 
      # initialize variables and run session         
      init = tf.global_variables_initializer()         
      sess = tf.Session()         
      sess.run(init)         
      # do training for numEpochs         
      for i in range(self.numEpochs):             
         for j in range(0,int(120/batchSize)):                 
            res = sess.run([loss,train], feed_dict={X: self.Xtrain[j*batchSize:(j+1)*batchSize], y: self.ytrain[j*batchSize:(j+1)*batchSize]})                 
            if(j == int(120/batchSize)-1):                     
               self.lossPlotData[hidden_node].append(res[0])                 
            self.W1[hidden_node] = sess.run(W1)                 
            self.W2[hidden_node] = sess.run(W2)                 
            self.b1[hidden_node] = sess.run(b1)                 
            self.b2[hidden_node] = sess.run(b2)            
      writer = tf.summary.FileWriter("myoutput", sess.graph)         
      writer.close()         
      print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_node, self.numEpochs, self.lossPlotData[hidden_node][-1]))         
      sess.close() 
 
 