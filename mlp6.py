
import numpy as np

np.random.seed(1)

class mlp():
    def __init__(self, inputs, targets, nhidden):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.eta = 0.4
        self.beta = 1.0
        self.nvectors = inputs.shape[0]
        self.ntargets = targets.shape[1]
        self.ninputs = inputs.shape[1]
        self.hiddenacc = np.zeros(self.nhidden)
        self.output = np.zeros(targets.shape[1])
        self.v = np.random.randn(*(len(self.inputs[0,:]) + 1, self.nhidden))*0.1
        self.w = np.random.randn(*(self.nhidden + 1, len(self.targets[0, :])))*0.1

    def sigmoid(self, x):
        return 1./(1 + np.exp(-self.beta*x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def include_bias(self, array):
        bias = -1
        return np.append(array, bias)

    def forward(self, inputs):
        """
        Takes in a single vector. 
        """
        inputs_tot = self.include_bias(inputs)
        h_chi = np.zeros(self.nhidden)
        a_chi = np.zeros(self.nhidden)
        h_kappa = np.zeros(self.ntargets)
        y_kappa = np.zeros(self.ntargets)
        #print(inputs_tot.shape, h_chi.shape, self.v.shape)
        for j in range(self.nhidden):
            for i in range(inputs.shape[0] + 1):
                h_chi[j] += inputs_tot[i] * self.v[i, j]           #With bias
            a_chi[j] = self.sigmoid(h_chi[j])
            self.hiddenacc[j] = a_chi[j]                           #no bias
        a_chi_tot = self.include_bias(a_chi)                       #With bias

        for i in range(self.ntargets):
            for j in range(self.nhidden + 1):
                h_kappa[i] += a_chi_tot[j] * self.w[j, i]          #With bias
            y_kappa[i] = self.sigmoid(h_kappa[i])
            self.output[i] = y_kappa[i]
        return y_kappa

    def backward(self, inputs, targets):
        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))
        w_T = self.w.T 
        delO = np.zeros(self.ntargets)
        delH = np.zeros(self.nhidden)

        hiddenacc_T = self.hiddenacc.T
        for k in range(self.ntargets):
            delO[k] = (self.output[k] - targets[k])*self.sigmoid_derivative(self.output[k])                
            for j in range(self.nhidden):
                updateW[j, k] = -self.eta*(hiddenacc_T[j]*delO[k])

        inputs_T = inputs.T
        for k in range(self.nhidden):            
            delH[k] = self.hiddenacc[k]*(1.0 - self.hiddenacc[k])*(sum(delO[:]*w_T[:, k]))                
            for i in range(inputs.shape[0]): 
                updateV[i, k] = -self.eta*(inputs_T[i]*delH[k])

        #updateV and updateW are one smaller than self.v and self.w because of bias 
        self.v += updateV
        self.w += updateW

    def train(self, inputs, targets):
        for n in range(inputs.shape[0]):
            self.forward(inputs[n])
            self.backward(inputs[n], targets[n])

    def error(self, validationset, validationstargets):
        error = np.zeros(validationset.shape[0])
        for i in range(validationstargets.shape[0]):
            validation_out = self.forward(validationset[i])
            error[i] = np.linalg.norm(validation_out - validationstargets[i])**2
        return sum(error)

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        epochs = 400
        count = 0
        error = np.zeros(epochs)
        epochs_final = 0
        for i in range(epochs - 1):
            self.train(inputs, targets)
            error[i] = self.error(validationset, validationstargets)       
            if error[i - 1] < error[i]:
                count += 1
            else:
                count = 0

            if count == 10:
                print('Error increasing %d times in a row. STOP' % count)
                print('Final epoch is:', i)
                epochs_final = i
                indices = np.linspace(i + 1, epochs, epochs - i)
                error = np.delete(error, indices)
                break
        return error, epochs_final

    def confusion_matrix(self, testset, testtargets):
        confusionmat = np.zeros((testtargets.shape[1], testtargets.shape[1]))		
        accuracy = 0
        for n in range(testtargets.shape[0]):
            predictedoutput = self.forward(testset[n])
            pred = np.argmax(predictedoutput)
            true = np.argmax(testtargets[n]) 
            confusionmat[pred, true] += 1
            if pred == true:
                accuracy += 1
        accuracy = accuracy/testtargets.shape[0]*100
        print('--------------------------------------------------------------------------------------------')
        print(f'Accuracy is found to be {accuracy} % with {self.nhidden} hidden nodes in hidden layer')
        print('--------------------------------------------------------------------------------------------') 
        return accuracy

    def k_fold(self, inputs_total, targets_total, Numbfolds):
        dataset = np.split(inputs_total[:-7], Numbfolds)
        dataset = np.array(dataset)
        targetset = np.split(targets_total[:-7], Numbfolds)
        targetset = np.array(targetset)
        #Holding out data to test on
        test = dataset[0]
        test_target = targetset[0]
        
        for i in range(dataset.shape[1]*Numbfolds, inputs_total.shape[0]):
            test = np.vstack((test, inputs_total[i,:]))
            test_target = np.vstack((test_target, targets_total[i,:]))
        #TRAINING AND VALIDATIONSET REMAIN, K-1 folds:
        train_and_test = np.delete(dataset, 0, 0)
        train_and_test_targets = np.delete(targetset, 0, 0)
        #print(train_and_test.shape)
        #print(train_and_test_targets.shape, np.delete(train_and_test_targets, 0 , 0).shape )
        valid = []
        train = []
        valid_targets = []
        train_targets = []
        for k in range(Numbfolds - 1):
            valid.append(train_and_test[k])
            valid_targets.append(train_and_test_targets[k])

            train.append(np.delete(train_and_test, k, 0))
            train_targets.append(np.delete(train_and_test_targets, k , 0))

            train_and_test = train_and_test
            train_and_test_targets = train_and_test_targets
           
        valid = np.array(valid)
        valid_targets = np.array(valid_targets)
        
        train = np.array(train) 
        train_targets = np.array(train_targets)
        train = np.reshape(train, (Numbfolds - 1,(Numbfolds - 2)*train_and_test.shape[1], 41))
        train_targets = np.reshape(train_targets, (Numbfolds-1, (Numbfolds - 2)*train_and_test.shape[1], 8))
        return train, train_targets, valid, valid_targets, test, test_target