"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import matplotlib.pyplot as plt

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0
        self.epochs = 100
        self.n_vectors = self.inputs.shape[0]                                                       #224 vectors
        self.vec_length = self.inputs.shape[1]                                                      #40 input nodes
        self.n_targets = self.targets.shape[1]       
        self.output_vectors = np.zeros((self.n_vectors, self.n_targets))                                               #8 output nodes
        #self.v = np.random.uniform(-0.3, 0.3, size = (len(self.inputs[0,:]) + 1, self.nhidden))     #40 weights + bias weights. Hidden layer        
        #self.w = np.random.uniform(-0.7, 0.7, size = (self.nhidden + 1, len(self.targets[0, :])))   #8 weights for output and a addtional bias weights. Output layer
        self.v = np.random.randn(*(len(self.inputs[0,:]) + 1, self.nhidden))*0.01
        self.w = np.random.randn(*(self.nhidden + 1, len(self.targets[0, :])))*0.01

    # You should add your own methods as well!
    def Sigmoid(self, u):
        """
        Activation function. 
        """
        return 1./(1 + np.exp(-self.beta*u))
    
    def reshape(self, A):
        """
        Returns the transposed of an A or matrix, e.g N x M -> M x N.
        Making code more readable.
        """
        A_trans = np.zeros((A.shape[1], A.shape[0]))
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                A_trans[col, row] = A[row, col]
        return A_trans

    def forward(self, inputs):
        """
        This algorithm feeds the inputs into the neural network. 
        Calculates y_k which is output. This is used to compute error
        The same notation is used from Marsland (An algorithmic persepctive 2nd ed) on Multilayer perceptron section. 
        """
        n_vectors = inputs.shape[0]
        z = np.zeros((n_vectors, self.nhidden))
        h_layer = np.zeros((n_vectors, self.nhidden))

        biasw = -np.c_[np.ones(inputs.shape[0])]
        biasv = -np.c_[np.ones(inputs.shape[0])]
        
        inputs_tot = np.concatenate((inputs, biasw), axis = 1)                #(224,41)        
        acc_chi = np.concatenate((z, biasv), axis = 1)                        #(224, 2)  
        
        h = np.zeros((n_vectors, self.n_targets))                  #(224, 8)
        y_k = np.zeros((n_vectors, self.n_targets))                      #(224, 8)

        for n in range(n_vectors):
            for chi in range(self.nhidden):
                for i in range(self.vec_length + 1):
                    h_layer[n, chi] += inputs_tot[n, i]*self.v[i, chi]
                acc_chi[n, chi] = self.Sigmoid(h_layer[n, chi])      
          
            for kappa in range(self.n_targets):
                for j in range(self.nhidden + 1):
                    h[n, kappa] += acc_chi[n, j]*self.w[j, kappa]  
                y_k[n, kappa] = self.Sigmoid(h[n, kappa])
        y_k = self.output_vectors
        return y_k, acc_chi, inputs_tot

    def backward(self, output, targets, acc, input_init):
        """
        calculating error from output after the inputs are fed forward 
        from the network. movements.py calls this function. Targets are the training targets. 
        So called backward propagation algo.
        """

        #error in output
        n_vectors = input_init.shape[0]
        delO = np.zeros((n_vectors, self.n_targets))
        for n in range(n_vectors):
            for kappa in range(self.n_targets):
                delO[n, kappa] = (output[n, kappa] - targets[n, kappa])*output[n, kappa]*(1 - output[n, kappa]) 
        
        delO_T = self.reshape(delO)
        #error in hidden layer
        delH = np.zeros((n_vectors, self.nhidden))
        for n in range(n_vectors):
            for chi in range(self.nhidden):
                delH[n, chi] = acc[n, chi]*(1 - acc[n, chi])*sum(self.w[chi, :]*delO_T[:, n]) 
        
        inputs_T = self.reshape(input_init)
        update_v = np.zeros((input_init.shape[1], delH.shape[1]))
        
        for k in range(input_init.shape[1]):
            for j in range(self.nhidden):
                for n in range(n_vectors):
                    update_v[k, j] = self.eta*inputs_T[k, n]*delH[n, j]
        self.v +=  update_v
  
        acc_T = self.reshape(acc)
        update_w = np.zeros((acc.shape[1], delO.shape[1]))
        
        for i in range(acc.shape[1]):
            for j in range(self.n_targets):
                for n in range(n_vectors):
                    update_w[i, j] = self.eta*acc_T[i, n]*delO[n,j]
        self.w +=  update_w

    def train(self, inputs, targets):
        """
        Running a number of epochs updating the weights for each epoch. 
        Calling forward to get output, then output into backward prop.
        """
        Numb_error = np.zeros(self.epochs)
        #print(inputs.shape)
        for i in range(self.epochs):
            output, acc, input_init = self.forward(inputs)
            self.backward(output, targets, acc, input_init)
            Numb_error[i] = np.linalg.norm(targets - output)**2
        return Numb_error/inputs.shape[0]

    def errorval(self, validation_set, validation_target):
        """
        Calculate error for validation data. 
        Use this to find a trend in the data and stop when starting to overfit
        """
        error_val = self.train(validation_set, validation_target)        
        return error_val

    def earlystopping(self, inputs, targets, valid, validtargets):
        """
        Want to see when model begins to overfit 
        This is done on validation set 
        """
        count = 0
        afterstopping_val = []
        beforestopping_val = []
        afterstopping_train = []
        beforestopping_train = []
        epochgood = []
        epochbad = []
        error_train = self.train(inputs, targets)
        error = self.errorval(valid, validtargets)
        #print(error_train)

        for i in range(self.epochs - 1):
            #error_train = self.train(inputs, targets)

            if error_train[i] < error_train[i + 1]:
                count += 1
                afterstopping_val.append(error[i])
                afterstopping_train.append(error_train[i])
                epochbad.append(i)
                #print('starting to overfit, error increasing')
            else:
                count = 0
                epochgood.append(i)
                beforestopping_val.append(error[i])
                beforestopping_train.append(error_train[i])
                #print('error decreasing') 
            if count == 10:
                print('increasing error happend %d times in a row at epoch %d' % (count, i-count))
                break
        return epochgood, epochbad, beforestopping_val, afterstopping_val, beforestopping_train, afterstopping_train, error_train, error, self.epochs

    def confusion(self, test_targets, test_set):
        confusion_mat = np.zeros((test_targets.shape[1], test_targets.shape[1]))
        pred, a_chi, inputss = self.forward(test_set)
        for n in range(test_set.shape[0]):    
            expected = np.argmax(test_targets[n])
            predicted = np.argmax(pred[n])
            confusion_mat[expected, predicted] += 1        
        print(confusion_mat)
