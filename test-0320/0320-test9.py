from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
import math
from scipy import sparse
from scipy.sparse import csr_matrix, save_npz, load_npz


num_newsgroups = 20
num_training = 12000
num_tests = 6774
#num_tests = 10
#num_training = 50
vocab_size = 61188
lamda=0.001
def get_parsed_matrix(csv_file, matrix_file):
    """Parses the data out of the data file and into a format used by naive bayes.

    :rtype: ndarray
    :returns: A matrix read from the csv file
    """
    matrix = None

    # Check to see if we have a saved npz file first
    if os.path.isfile(matrix_file):
        sparse_matrix = load_npz(matrix_file)
        matrix = sparse_matrix.todense()
    else:
        # initialize the matrix with 0's. This will help speed up the time to parse the data file
        if 'testing' in csv_file.name:
            matrix = np.zeros((num_tests, vocab_size+1), dtype=np.float64)
        
        else:
            matrix = np.zeros((num_training, vocab_size+2), dtype=np.float64)
        row = 0
        for line in csv_file.readlines():
            matrix[row, :]= list(map(int,line.split(',')))
            row += 1
       
        # Gets rid of the first column of ids. We don't need this
        # since that information is based on the row. i.e. row 0 is ID 1.
        #matrix = matrix[:, 1:]
        matrix[:,0]=1
        # save a sparse version of the matrix to reduce size and speed up reading time
        sparse_matrix = csr_matrix(matrix)
        save_npz(matrix_file, sparse_matrix)
        
    # returns a normal matrix. Sparse matrices don't have the same indexing power
    # as normal matrices, so we will be using normal matrices in the other functions.
    
    return matrix

def initialize_betas(dim):
    w = np.random.rand(dim)
    return w

def cost_func(weights, X, Y):
    """Logistic loss, numerically stable implementation.
    
    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    loss: float
    """
    z = np.matmul(X,weights)
    z=np.asarray(z).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    return np.mean((1 - Y) * z - logsig(z))

def logsig(x):
    """Compute the log-sigmoid function component-wise."""
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out



def train(X_features, Y_labels, weights, lr, iters):
    cost_history = []
    a=[]
    cost=0
    i=0
    #for i in range(1000):
        #a.append(i)
    for i in range(iters):
        weights = update_weights(X_features, Y_labels, weights, lr)
        
        #Calculate error for auditing purposes
        #cost = cost_func(weights,X_features, Y_labels)
        #cost_history.append(cost)

        # Log Progress
        #if i % 2 == 0:
            #print ("iter: " +str(i) + " cost: "+str(cost))
    #plt.plot(a,cost_history)
    #plt.show()
    #print(weights)
    return weights, cost_history

def getbeta_forclasses(features,labels,weights):
    classlabel=1
    num=vocab_size+1
    #initial_weight=initialize_betas(X.shape[1])
    result_beta_matrix=np.zeros([num,1])
    for classlabel in range (1,21):
        y_new=np.where(labels==classlabel,1,0)
        weights,cost=train(features,y_new,weights,0.001,10000)
        #weights=weights.reshape(-1,1)
        result_beta_matrix=np.concatenate((result_beta_matrix,weights),axis=1)
    result_beta_matrix=np.delete(result_beta_matrix,0,1)
    
    return result_beta_matrix


    
def get_classification(X,result_beta_matrix):
    result_matrix=np.matmul(X,result_beta_matrix)
    #print(result_matrix)
    return np.argmax(result_matrix,axis=1)

def classify_data(result_array):
        """Find the classification for the given testing data.

        :type data: list
        :param data: Parsed testing data to classify.

        :rtype: list
        :returns: A classification for each data item in the testing data set.
        """
        classification = []
        i=12001
        result_array=np.asarray(result_array).reshape(-1)
        for item in result_array:
            
           
            
            classification.append({'id': i,
                                   'class': 1+result_array[i-12001]})
            i=i+1
        return classification

def save_result_File(classification, result_file):
    """Saves the classification from the ID3 algorithm to a file.

    :type classification: list
    :param classification: The classification output from the ID3 algorithm for the testing data.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    print('id', 'class', file=result_file, sep=',')
    for item in classification:
        idx, cls = item.values()
        print(idx, cls, file=result_file, sep=',')
    return






def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x,dtype=np.float64)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def f_gradient(weights, training_features, training_labels):
    """Computes the gradient of the logistic loss.
    
    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    grad: array-like, shape (n_features,)    
    """
    
    #straining_features=sparse.csr_matrix(training_features)
    #straining_labels = sparse.csr_matrix(training_labels)
    #Z=straining_features*(sparse.csr_matrix(weights))
    #z=Z.todense()
    #z = A.dot(x)
    #z = np.asarray(z)
    c=np.matmul(training_features,weights)
    c=np.squeeze(c)
    slabels=np.squeeze(training_labels)
    s=expit_b(c,slabels)
    s=np.array(s).reshape(-1,1)
    t_features=training_features.transpose()
    result=np.matmul(t_features,s)
    #training_labels= np.asarray(training_labels)
    #s = expit_b(z,training_labels)
    #print('sshape is')
    #print(s.shape)
    #number=training_features.shape[0]
    #result=straining_features.transpose()*(sparse.csr_matrix(s))
    return result 

def update_weights(features, labels, weights, lr):
    #N = len(features)

    
    

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = f_gradient(weights,features,labels)

    #3 Take the average cost derivative for each feature
    #gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr
    small_weights=lr*lamda*weights
    #5 - Subtract from our weights to minimize cost
    #weights=np.array(weights).reshape(-1,1)
    #weights -= gradient
    weights=weights-gradient-small_weights
    return weights

def split(X_total):
     
     X = np.delete(X_total, -1, axis=1)
     Y=np.array(X_total[:,-1])
     X_total
     X[:,0]=1
     X = np.array(X)
     Y=np.array(Y).reshape(-1,1)
     return X,Y


resultfile=open('training.csv','r')
#matrixfile=open('matrix','w')
testingfile=open('testing.csv','r')

X_total=get_parsed_matrix(resultfile,'training-matrix-total.npz')
X_test=get_parsed_matrix(testingfile,'testing-matrix-total.npz')
X_test[:,0]=1
X_test = np.array(X_test)
features,labels=split(X_total)
weights_initial=initialize_betas(61189)
weights_initial=np.array(weights_initial).reshape(-1,1)

beta=getbeta_forclasses(features,labels,weights_initial)

result_array=get_classification(X_test,beta)

classification=classify_data(result_array)
resultfile1=open('0320-test9.csv','w')
save_result_File(classification, resultfile1)
resultfile.close()
resultfile1.close()
