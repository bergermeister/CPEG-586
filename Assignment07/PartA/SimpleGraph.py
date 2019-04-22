import tensorflow as tf 
 
class SimpleGraph(object): 
   def simpleComputation(self):
      a = tf.placeholder("float", name='pholdA')  
      b = tf.Variable(2.0, name='varB')          
      b2 = tf.Variable(b+2,name='varB2')           
      init = tf.global_variables_initializer() 
 
      d = tf.Variable(0.0, name='varD')         
      # Define a constant         
      c = tf.constant([1., 2., 3., 6.], name='consC')           
      print("c:", c)           
      d = a * b + c + b2         
      with tf.Session() as sess:             
         sess.run(init)              
         # we need to write the graph to a file so that tensorboard can 
         # launched. Type the following command in shell             
         # tensorboard --logdir=D:\DeepLearning\TensorFlowApps\GraphApi1\GraphApi1\myoutput             
         # then view tensor board via the browser             
         writer = tf.summary.FileWriter("myoutput", sess.graph)      
         
         # Run a session and calculate d             
         # # compute the d node, feed a with values of 0.5, 3, 5             
         # print(sess.run(d, feed_dict={a: [[0.5],[3],[5]]}))             
         dres = sess.run(d, feed_dict={a: [[0.5]]})             
         print(dres)             
         writer.close()             
         sess.close() 
 
