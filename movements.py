#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp2
import matplotlib.pyplot as plt
np.random.seed(0)
filename = 'data/movements_day1-3.dat'

movements = np.loadtxt(filename, delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]
 
# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8))
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1


# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]
# Split data into 3 sets

# Training updates the weights of the network and thus improves the network
train = movements[::2,0:40]
train_targets = target[::2]
# Validation checks how well the network is performing and when to stop
valid = movements[1::4,0:40]
valid_targets = target[1::4]
# Test data is used to evaluate how good the completely trained network is.
test = movements[3::4,0:40]
test_targets = target[3::4]


# Try networks with different number of hidden nodes:
hidden = 6

# Initialize the network:
net = mlp2.mlp(train, train_targets, hidden)
# Run training:
# NOTE: You can also call train method from here,
#       and make train use earlystopping method.
#       This is a matter of preference.

# Check how well the network performed:


outputsFed, a_chi, inputs = net.forward(train)
epochgood, epochbad, bforestop_val, aftrstop_val, bforestop_train, aftrstop_train, err_train, err, epochs = net.earlystopping(train, train_targets, valid, valid_targets)

#error_val = net.errorval(valid, valid_targets)
net.confusion(test_targets, test)

#plotting
epochs = np.linspace(0, epochs - 1, epochs)  
##plt.scatter(epochgood, bforestop_val, color = 'b',marker = 'o', label = 'Validation set: Training MLP') 
##plt.scatter(epochbad, aftrstop_val, color = 'b', marker = '*', label = 'Validation set:After training MLP')
##plt.scatter(epochgood, bforestop_train, color = 'r',marker = '^', label = 'Training MLP') 
##plt.scatter(epochbad, aftrstop_train, color = 'r', marker = 's', label = 'After training MLP')
plt.plot(epochs, err_train, 'r', label = 'Training')
plt.plot(epochs, err, 'b', label = 'Validation')
plt.xlabel(r'Number of epochs', fontsize = 16)
plt.ylabel(r'Error', fontsize = 16)
plt.legend()
plt.show()
