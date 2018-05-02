import numpy as np
import  h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import *
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    w1=np.random.randn(n_h,n_x)*0.01
    w2=np.random.randn(n_y,n_h)*0.01
    b1=np.zeros((n_h,1))
    b2=np.zeros((n_y,1))
    assert (w1.shape==(n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (w2.shape == (n_y,n_h))
    assert (b2.shape == (n_y,1))
    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    L=len(layer_dims)
    for i in range(1,L):
        parameters['w'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
        assert (parameters['w' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters
def linear_forward(A,w,b):
    Z=np.dot(w,A)+b
    assert (Z.shape==(w.shape[0],A.shape[1]))
    cache=(A,w,b)
    return Z,cache

def linear_activation_forword(A_prev,w,b,activation):
    if activation=='sigmoid':
        Z,linear_cache=linear_forward(A_prev,w,b)
        A,activation_cache=sigmoid(Z)
    elif activation=='relu':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A,activation_cache = relu(Z)
    assert (A.shape == (w.shape[0], A_prev.shape[1]))
    cache=(linear_cache,activation_cache)
    return  A, cache

def L_model_forward(X,parameters):
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prev=A
        A, cache = linear_activation_forword(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache =linear_activation_forword(A, parameters['w' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return  AL,caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost=np.squeeze(cost)
    assert (cost.shape==())
    return cost

def linear_backward(dZ,cache):
    A_prev,w,b=cache
    m=A_prev.shape[1]
    dw=1/m*np.dot(dZ,A_prev.T)
    db=1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(w.T,dZ)
    assert(dA_prev.shape==A_prev.shape)
    assert (dw.shape==w.shape)
    assert (db.shape==b.shape)
    return dA_prev,dw,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache=cache
    if activation=='relu':
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dw,db=linear_backward(dZ,linear_cache)
    elif activation=='sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dw, db = linear_backward(dZ, linear_cache)
    return dA_prev,dw,db

def L_model_backward(AL,Y,caches):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL=(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache=caches[L-1]
    grads["dA" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,'sigmoid')
    for l in reversed((range(L-1))):
        current_cache=caches[l]
        dA_prev_temp,dw_temp,db_temp=linear_activation_backward(grads["dA" + str(l+2)], current_cache, 'relu')
        grads['dA'+str(l+1)]=dA_prev_temp
        grads["dw" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)
    for l in range(L):
        parameters['w'+ str(l+1)] = parameters['w' + str(l + 1)] - learning_rate * grads['dw' + str(l + 1)]
        parameters['b'+ str(l+1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters

def two_layer_model(X,Y,layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads={}
    costs=[]
    m=X.shape[1]
    (n_x,n_h,n_y)=layers_dims
    parameters=initialize_parameters(n_x,n_h,n_y)
    w1=parameters['w1']
    w2=parameters['w2']
    b1=parameters['b1']
    b2=parameters['b2']
    for i in range(0,num_iterations):
        A1,cache1=linear_activation_forword(X,w1,b1,'relu')
        A2,cache2=linear_activation_forword(A1,w2,b2,'sigmoid')
        cost=compute_cost(A2,Y)
        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))
        dA1, dw2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dw1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        grads['dw1']=dw1
        grads['db1']=db1
        grads['dw2']=dw2
        grads['db2']=db2
        parameters = update_parameters(parameters, grads, learning_rate)
        w1=parameters['w1']
        w2=parameters['w2']
        b1=parameters['b1']
        b2=parameters['b2']
        if print_cost and i%100==0:
            print('cost after interation{}:{}'.format(i,np.squeeze(cost)))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs=[]
    parameters=initialize_parameters_deep(layers_dims)
    for i in range(0,num_iterations):
        AL,caches=L_model_forward(X,parameters)
        cost=compute_cost(AL,Y)
        grads=L_model_backward(AL,Y,caches)
        parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i%100==0:
            print('cost after iteration %i:%f'%(i,cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.xlabel('interation ')
    plt.ylabel('cost')
    plt.title('learning rate='+str(learning_rate))
    plt.show()
    return parameters

num_px=64
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
train_x_orig,train_y,test_x_orig,test_y,classes=load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
#parameters=two_layer_model(train_x,train_y,layers_dims,learning_rate = 0.0075, num_iterations = 3000, print_cost=True)
#parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


