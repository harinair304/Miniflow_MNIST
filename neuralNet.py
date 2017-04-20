# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:03:27 2017

@author: harinair
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from miniflow import *
from sklearn.utils import resample


training_data = pd.read_csv('data//train.csv')
test_data = pd.read_csv('data//test.csv')




train_labels = training_data.iloc[:,0].as_matrix()
train_pixels = training_data.iloc[:, 1:785].as_matrix().astype(float)
train_pixels = (train_pixels)/255



#train_labels.hist()
#
#train_im = training_data.iloc[3000, 1:785]
#
#image = train_im.values.reshape((28,28))
#
#plt.imshow(image)
#

n_features = train_pixels.shape[1]
n_hidden = 400
W1_ = np.random.uniform(low = -1, high=1, size=(n_features, n_hidden))
b1_ = np.zeros(n_hidden)
W2_ = np.random.uniform(low = -1, high=1, size=(n_hidden, 10))
b2_ = np.zeros(10)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

#l1 = Linear(X, W1, b1)
#s1 = Sigmoid(l1)
#l2 = Linear(s1, W2, b2)
#cost = MSE(y, l2)

####The network architecture##########

l1 = Linear(X, W1, b1)
r1 = ReLU(l1)
l2 = Linear(r1,W2, b2)
sfmax = Softmax(l2,y)
cost = CrossEnt(sfmax,y)
 
#######################################

feed_dict = {
    X: train_pixels,
    y: train_labels,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

i=0
j=0
epochs = 25

m = train_pixels.shape[0]
batch_size = 256
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
forward(graph)

W1_op=W1.value
W2_op=W2.value
l1_output = l1.value
r1_output = r1.value
l2_output = l2.value
sfmax_value = sfmax.value

trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(train_pixels, train_labels, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
    

#plotting weights. This didn't show anything interesting. Probably because there are two weight system, and taking the dot product to plot them doesn't make sense (?) 

Wtot = np.dot(W1.value,W2.value)

w_min = np.min(Wtot)
w_max = np.max(Wtot)


fig, axes = plt.subplots(3, 4)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = Wtot[:, i].reshape((28,28))

            # Set the label for the sub-plot
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.matshow(image, cmap=plt.cm.gray, vmin=0.5*w_min,
               vmax=.5*w_max)

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])



#Predictions

test_labels = test_data.iloc[:,0].as_matrix()
test_pixels1 = test_data.as_matrix().astype(float)
test_pixels1 = (test_pixels1)/255


feed_dict4 = {
    X: test_pixels1,
    y: test_labels,
    W1: W1.value,
    b1: b1.value,
    W2: W2.value,
    b2: b2.value
}

graph_pred3 = topological_sort(feed_dict4)

forward(graph_pred3)

y_pred_prob = sfmax.value

y_pred = np.argmax(y_pred_prob,axis =1)


#Helper code for creating the submission file
import csv
header = ['ImageID','Label']
with open('sumbission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter = ',')
    writer.writerow(header)
    for i, p in enumerate(y_pred.T):
        writer.writerow([str(i+1), str(p)])