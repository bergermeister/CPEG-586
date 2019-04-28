import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
 
class NNGraph1(object):     
   def __init__(self):         
      self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.readTrainTestData()         
      # model parameters for three architectures         
      self.num_hidden_nodes = 10          
      self.lossPlotData = []           
      self.W1 = np.random.rand(4, self.num_hidden_nodes)           
      self.W2 = np.random.rand(self.num_hidden_nodes, 3) # 3 outputs         
      self.b1 = np.random.rand(self.num_hidden_nodes)           
      self.b2 = np.random.rand(3)          
      self.numEpochs = 500 
 
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
 
   def plotLoss(self):         
      plt.figure(figsize=(12,8))         
      plt.plot(range(self.numEpochs), self.lossPlotData, label="nn: 4-%d-3" % self.num_hidden_nodes)         
      plt.xlabel('Iteration', fontsize=12)           
      plt.ylabel('Loss', fontsize=12)          
      plt.legend(fontsize=12)           
      plt.show() 
 
   def trainAndTestArchitecture(self):         
      # training is done inside a tensorflow session         
      self.W1, self.W2,self.b1,self.b2 = self.trainNNmodel()                  
      # session is closed so plotting is done outside of tensorflow's session         
      self.plotLoss() 
 
      # another tensorflow session is created to compute accuracies on test data         
      self.computeAccuracy() 
 
   def computeAccuracy(self):         
      # Evaluate models on the test set         
      X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')           
      y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y') 
 
      # evaluate the network based on trained weights         
      W1 = tf.Variable(self.W1)         
      W2 = tf.Variable(self.W2)         
      b1 = tf.Variable(self.b1)         
      b2 = tf.Variable(self.b2)         
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
      print('Network architecture 4-%d-3, accuracy: %.2f%%' % (self.num_hidden_nodes, accuracy)) 
 
   def trainNNmodel(self):         
      tf.reset_default_graph() # reset the graph as this will be called multiple times                                  
      # for different models (numer of hidden nodes)         
      # # placeholders for input and expected outputs         
      batchSize = 10         
      X = tf.placeholder(shape=(batchSize, 4), dtype=tf.float64, name='X')        
      y = tf.placeholder(shape=(batchSize, 3), dtype=tf.float64, name='y') 
 
      # create tf variables for weight matrices and biases         
      W1 = tf.Variable(self.W1, dtype=tf.float64)         
      W2 = tf.Variable(self.W2, dtype=tf.float64)         
      b1 = tf.Variable(self.b1, dtype=tf.float64)        
      b2 = tf.Variable(self.b2, dtype=tf.float64)         
      # neural net graph         
      A1 = tf.sigmoid(tf.add(tf.matmul(X, W1),b1))         
      A2 = tf.sigmoid(tf.add(tf.matmul(A1, W2),b2))         
      #A2 = tf.nn.softmax(tf.matmul(A1, W2) + b2)  # use with cross entropy loss 
 
      # define loss          
      error = tf.square(A2 - y)         
      loss = tf.reduce_sum(error)         
      #cross_entropy_error = -tf.reduce_sum(y*tf.log(A2 + 1e-10))         
      #loss = tf.reduce_sum(cross_entropy_error) 
 
      # define train function to minimize the loss         
      optimizer = tf.train.GradientDescentOptimizer(0.05)         
      #optimizer = tf.train.AdamOptimizer(learning_rate=0.1) # the optimizer 

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
               #lossPlotData[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain[j*batchSize:(j+1)*batchSize], y: ytrain[j*batchSize:(j+1)*batchSize]}))                     
               self.lossPlotData.append(res[0])                 
            weights1 = sess.run(W1) # Note w1,w2,b1,b2 do not depend on a                 
            weights2 = sess.run(W2) # previous node, so sess.run does not                 
            bias1 = sess.run(b1)    # cause whole graph to be computed again                 
            bias2 = sess.run(b2)                
      writer = tf.summary.FileWriter("myoutput", sess.graph)         
      writer.close()         
      print("loss (hidden nodes: %d, iterations: %d): %.2f" % (self.num_hidden_nodes, self.numEpochs, self.lossPlotData[-1]))         
      sess.close()         
      return weights1, weights2, bias1, bias2