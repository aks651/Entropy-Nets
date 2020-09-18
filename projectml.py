# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:29:53 2019

@author: Home1
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(45)
data= np.array(np.recfromcsv("heart.csv").tolist()).astype(float)

data[:, :-1] = (data[:, :-1] - np.mean(data[:, :-1], axis = 0))/np.std(data[:, :-1])
train,test=train_test_split(data,test_size=0.10)

X_train = train[:, :-1]
Y_train=train[:, -1]
X_test=test[:, :-1]
Y_test=test[:, -1]
y_unique,y_counts=np.unique(Y_train,return_counts=True)


estimator=DecisionTreeClassifier(max_depth=3,random_state=0)
estimator.fit(X_train,Y_train)

n_nodes=estimator.tree_.node_count
children_left=estimator.tree_.children_left
children_right=estimator.tree_.children_right
feature=estimator.tree_.feature
threshold=estimator.tree_.threshold

node_depth=np.zeros(shape=n_nodes,dtype=np.int64)
is_leaves=np.zeros(shape=n_nodes,dtype=bool)
stack=[(0,-1)]
while len(stack)>0:
    
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.w = np.array(np.recfromcsv("Matrix2.csv", names = None).tolist())
        self.weights[1] = self.weights[1]*self.w
#    
    def SGD(self, training_data, epochs = 1000, batch_size = 16, update_rate = 0.6):
        n = training_data.shape[1]
        print("Epoch :", end = " ")
        for j in range(epochs):
            print("%d," %(j+1), end = " ")
            np.random.shuffle(training_data.T)
            batches = [training_data[:, k:k+batch_size] for k in range(0, n, batch_size)]
            for data in batches:
                self.update_batch(data.T, update_rate)   
        print(" ")
    
    def update_batch(self, data, update_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(data[:, :-1], data[:, -1]):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (2*update_rate/len(data))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(2*update_rate/len(data))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights[1] = self.weights[1]*self.w
    
    def backprop(self, x, y):
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        outputs = [np.array(x, ndmin = 2).T]
        for b, w in zip(self.biases, self.weights):
            outputs.append(sigmoid(np.dot(w, outputs[-1])+b))
        # backward pass
        error = (outputs[-1] - y) * outputs[-1] * (1 - outputs[-1])
        del_b[-1] = error.T
        del_w[-1] = np.dot(error, outputs[-2].T)
        for i in range(2, self.num_layers):
            error = np.dot(self.weights[-i+1].T, error) * outputs[-i] * (1 - outputs[-i])
            del_b[-i] = error
            del_w[-i] = np.dot(error, outputs[-i-1].T)
        return(del_b, del_w)
    
    def evaluate(self, x, y):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
#        y_pred = x
        index = np.argwhere(x[0] > 0.5).T[0]    
        TP = len(np.argwhere(y[index] == 1).T[0])
        FP = len(index) - TP
        index = np.argwhere(x[0] < 0.5).T[0]
        TN = len(np.argwhere(y[index] == 0).T[0])
        FN = len(index) - TN
        
#        print(y_pred)
#        print(y)
        precision_A = TP/(TP+FP)
        precision_B = TN/(TN+FN)
        recall_A = TP/(TP+FN)
        recall_B = TN/(TN+FP)
        F1_A = np.round(2*precision_A*recall_A/(precision_A + recall_A),2)
        F1_B = np.round(2*precision_B*recall_B/(precision_B + recall_B),2)
        print("F1 SCORES")
        print("Positive class : %0.2f" %F1_A)
        print("Negative class : %0.2f" %F1_B)
    
def sigmoid(u):
        return(1/(1 + np.exp(-u)))
        
net = Network([train.shape[1] - 1, 7,8, 1])
net.SGD(train.T)
net.evaluate(X_test.T, Y_test)