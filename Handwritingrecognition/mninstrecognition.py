#-*-conding:utf-8-*-
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/Handwritingrecongnition/',one_hot= True)
#print(mnist.train.num_examples,mnist.validation.num_examples,mnist.test.num_examples)
#plt.imshow(mnist.train.images[1].reshape(28,28))
batch_size=100
#xs,ys=mnist.train.next_batch(batch_size)
input_node=784
out_node=10
layer1_node=500
learning_rate_base=0.8
learning_rate_decay=0.99
regularization_rate=0.0001
train_steps=30000
moving_average_decay=0.99
def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        z2=tf.matmul(layer1,weight2)+biases2
        return z2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        z2=tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)
        return z2
def train(mnist):
    x=tf.placeholder(tf.float32,[None,input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,out_node],name='y-input')
    weight1=tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[layer1_node]))
    weight2=tf.Variable(tf.truncated_normal([layer1_node,out_node],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[out_node]))
    y=inference(x,None,weight1,biases1,weight2,biases2)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    loss1=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(loss1)
    regulaizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    regulaization=regulaizer(weight1)+regulaizer(weight2)
    loss=cross_entropy_mean+regulaization
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)#在min中传入global_step将自动更新global_step参数。从而使得学习率更新
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        for i in range(train_steps):
            xs,ys=mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            if i %1000==0:
                train_acc=sess.run(accuracy,feed_dict={x:xs,y_:ys})
                print('after %d training step(s),train accutacy''using average model is %g'%(i,train_acc))
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print('after %d training step(s),validation accutacy''using average model is %g'%(i,validate_acc))
                
            
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('after %d training step(s),test accutacy''using average model is %g'%(i,test_acc))

train(mnist)
        

    
    
    
    