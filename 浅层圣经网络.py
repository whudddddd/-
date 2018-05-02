import  matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import sys
import os
import numpy as np
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from lr_utils import load_dataset

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def initializa_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    assert (w.shape==(dim,1))
    assert (isinstance(b,float))or(isinstance(b,int))
    return w,b
def propagate(w,b,X,Y):
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)
    ### END CODE HERE ###
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads,cost=propagate(w,b,X,Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w=w-learning_rate*dw
        b=b-learning_rate*db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A=sigmoid(np.dot(w.T,X)+b)
    ### END CODE HERE ###

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0, i] = 1
        ### END CODE HERE ###
    assert (Y_prediction.shape == (1, m))

    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b=initializa_with_zeros(X_train.shape[0])    # initialize parameters with zeros (≈ 1 line of code)
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# Example of a picture
'''index = 2
new_im = Image.fromarray(train_set_x_orig[index])
#new_im.show()
plt.imshow(train_set_x_orig[index])'''
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
'''print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))'''
test_set_x=test_set_x_flatten/255
train_set_x=train_set_x_flatten/255
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)
'''learning_rate=[0.01,0.001,0.0001]#选择合适的学习率
models={}
for i in learning_rate:
    print('learning rate is:'+str(i))
    models[str(i)]=model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=i, print_cost=False)
    print('\n'+'------------------------------------------------'+'\n')
for i  in learning_rate:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('interations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.8')#设置图例legend背景色
plt.show()'''
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image2.jpg"   # change this to the name of your image file
## END CODE HERE ##
# We preprocess the image to fit your algorithm.
fname = base_dir+"/images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T#放大缩小图片
my_predicted_image = predict(d["w"], d["b"], my_image)
print(my_image.shape)
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")