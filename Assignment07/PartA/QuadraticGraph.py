import tensorflow as tf
import math 
 
class QuadraticGraph(object):
   def computeRoot(self):
      tf.reset_default_graph()
      # r1 = (-b + sqrt(b*b - 4*a*c))/2*a, r2 = (-b - sqrt(b*b - 4*a*c))/2*a         
      # define placeholders         
      a = tf.placeholder("float", name='pa')           
      b = tf.placeholder("float", name='pb')           
      c = tf.placeholder("float", name='pc')         
      fourc = tf.constant(4.0, name='c4')         
      twoc = tf.constant(2.0, name='c2')         
      r1 = (-b + (b*b - fourc*a*c)**0.5)/twoc*a         
      r2 = (-b - (b*b - fourc*a*c)**0.5)/twoc*a  # math.sqrt does not work         
      roots = [r1,r2]           
      
      # Run a session and calculate roots         
      with tf.Session() as sess:             
         # since no tf variables are decared, call to             
         # tf.global_variables_initializer() is not needed 

         # tensorboard -logdir=D:\DeepLearning\TensorFlowApps\GraphApi1\GraphApi1\myoutput            
         # then view tensor board via the browser             
         writer = tf.summary.FileWriter("myoutput", sess.graph)             
         # compute the roots, feed a with values of 1,2,1   and 1,3,2             
         print(sess.run(roots, feed_dict={a: [1,1],b:[2,3],c:[1,2]}))             
         writer.close()             
         sess.close() 
 