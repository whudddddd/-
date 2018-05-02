# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:37:50 2018

@author: gemslab
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

eval_interval_secs=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_inference.input_node],name='x_input')
        y_=tf.placeholder(tf.float32,[None,mnist_inference.out_node],name='y_input')
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        y=mnist_inference.inference(x,None)
        correct_predicton=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)
        while   True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(checkpoint_dir='"D:/\347\250\213\345\272\217/python/dl/Handwritingrecognition/save//')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print('after %s trainning step(s),validate accuracy=%g'%(global_step,accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(eval_interval_secs)

def main(argv=None):
    mnist=input_data.read_data_sets('/mnist/',one_hot= True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()
    
                