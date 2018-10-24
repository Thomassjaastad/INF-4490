
import numpy as np

np.random.seed(1)

class mlp():
    def __init__(self, inputs, targets, nhidden):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.eta = 0.01
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

    def Recall(self, data):
        for n in range(data.shape[0]):
            out = self.forward(data[n])
            return out    
    
    def train(self, inputs, targets):
        #iterations = 100
        #for i in range(iterations):
        for n in range(inputs.shape[0]):
            self.forward(inputs[n])
            self.backward(inputs[n], targets[n])

    def error(self, validationset, validationstargets):
        error = np.zeros(validationset.shape[0])
        for i in range(validationstargets.shape[0]):
            validation_out = self.Recall(validationset)
            error[i] = np.linalg.norm(validation_out - validationstargets[i])**2
            #print(validation_out.shape, validationstargets[i].shape)
        return sum(error)

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        epochs = 100
        count = 0
        error = np.zeros(epochs)
        for i in range(epochs - 1):
            self.train(inputs, targets)
            error[i] = self.error(validationset, validationstargets)       
            if error[i - 1] < error[i]:
                count += 1
            else:
                count = 0
            if count == 10:
                print('Error increasing %d times in a row. STOP' % count)
                break
            print(error[i])

    def confusion_matrix(self, testset, testtargets):
        confusionmat = np.zeros((testtargets.shape[1], testtargets.shape[1]))		
        for n in range(testtargets.shape[0]):
            predictedoutput = self.Recall(testset)
            print(predictedoutput)
            pred = np.argmax(predictedoutput)
            true = np.argmax(testtargets[n]) 
            confusionmat[pred, true] += 1
        return confusionmat

