
import numpy as np



np.random.seed(1)
class mlp():
    """
    n_vectors = number of inputs for single node
    n_inputs = number of inputs nodes
    n_targets = number of output and targets nodes 
    inputs = (n_vectors, n_inputs)
    targets = (n_vectors, n_targets)
    """
    def __init__(self, inputs, targets, nhidden):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.eta = 0.1
        self.beta = 1.0
        self.nvectors = inputs.shape[0]
        self.ntargets = targets.shape[1]
        self.ninputs = inputs.shape[1]
        self.hiddenacc = np.zeros((self.nvectors, self.nhidden))
        self.output = np.zeros(targets.shape[1])
        self.v = np.random.randn(*(len(self.inputs[0,:]) + 1, self.nhidden))*0.1
        self.w = np.random.randn(*(self.nhidden + 1, len(self.targets[0, :])))*0.1

    def sigmoid(self, x):
        return 1./(1 + np.exp(-self.beta*x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def include_bias(self, array):
        bias = -np.c_[np.ones(array.shape[0])]
        return np.concatenate((array, bias), axis = 1)

    def transpose(self, A):
        """
        Transposes a matrix A
        """
        A_trans = np.zeros((A.shape[1], A.shape[0]))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_trans[j, i] = A[i, j]
        return A_trans

    def forward(self, inputs):
        """
        Need to fix bias on inputs. inputs values change.
        """

        #numb_vec = inputs.shape[0]
        #inputs_tot = self.include_bias(inputs)
        #print(inputs_tot.shape, self.v.shape)
        h_chi = np.zeros(self.nhidden)
        a_chi = np.zeros(self.nhidden)
        #a_chi_tot = self.include_biasw(inputs)
        h_kappa = np.zeros(self.ntargets)
        y_kappa = np.zeros(self.ntargets)
        print(h_chi.shape, inputs.shape, self.v.shape)
        exit()        
        #for n in range(numb_vec):
        for j in range(self.nhidden):
            for i in range(inputs.shape[1] + 1):
                h_chi[j] += inputs_tot[i] * self.v[i, j]           #With bias
            a_chi[j] = self.sigmoid(h_chi[j])
            self.hiddenacc[j] = a_chi[j]                           #no bias
        a_chi_tot = self.include_bias(a_chi)                       #With bias

        #for n in range(numb_vec):        
        for i in range(self.ntargets):
            for j in range(self.nhidden + 1):
                h_kappa[i] += a_chi_tot[j] * self.w[j, i]          #With bias
            y_kappa[n,i] = self.sigmoid(h_kappa[i])
            self.output[i] = y_kappa[i]
        return y_kappa


    #def deltaO(self, inputs, targets):
        #numb_vectors = inputs.shape[0]
        #delO = np.zeros((numb_vectors, self.ntargets))
        #for n in range(numb_vectors):
        #    for j in range(self.ntargets):
                
        #return delO

    #def deltaH(self, inputs, targets): 
        #numb_vectors = inputs.shape[0]
        #deltaOut = self.deltaO(inputs, targets)
        #delH = np.zeros((numb_vectors, self.nhidden))
        #w_T = self.transpose(self.w)
        #for n in range(numb_vectors):
        #    for k in range(self.nhidden):
                
        #return delH

    def backward(self, inputs, targets):
        numb_vectors = inputs.shape[0]
        #delH = self.deltaH(inputs, targets)
        #delO = self.deltaO(inputs, targets)

        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))
        w_T = self.transpose(self.w) 

        delO = np.zeros((numb_vectors, self.ntargets))
        delH = np.zeros((numb_vectors, self.nhidden))

        hiddenacc_T = self.transpose(self.hiddenacc)
        for n in range(numb_vectors):        
            for k in range(self.ntargets):
                delO[n, k] = (self.output[n, k] - targets[n, k])*self.sigmoid_derivative(self.output[n, k])                
                for j in range(self.nhidden):
                    updateW[j, k] = self.eta*(hiddenacc_T[j, n]*delO[n, k])


        inputs_T = self.transpose(inputs)
        for n in range(numb_vectors):        
            for k in range(self.nhidden):            
                delH[n, k] = self.hiddenacc[n, k]*(1.0 - self.hiddenacc[n, k])*(sum(delO[n, :]*w_T[:, k]))                
                for i in range(inputs.shape[1]): 
                    updateV[i, k] = self.eta*(inputs_T[i, n]*delH[n, k])



        #updateV and updateW are one smaller than self.v and self.w because of bias 
        self.v -= updateV
        self.w -= updateW

    def Recall(self, data):
        output = self.forward(data)
        return output
    
    def train(self, inputs, targets):
        epochs = 200
        for i in range(epochs):
            for n in range(inputs.shape[0]):
                self.forward(inputs[n])
                self.backward(inputs[n], targets[n])

    def error(self, validationset, validationstargets):
        numb_vectors = validationset.shape[0]
        error = np.zeros(validationset.shape[0])
        validation_out = self.Recall(validationset)
        for i in range(numb_vectors):
            error[i] = np.linalg.norm(validation_out[i] -  validationstargets[i])**2
            #print(error[i])
        return sum(error)

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        epochs = 200
        error = self.error(validationset, validationstargets)        
        count = 0
        for i in range(epochs):
            if error[i] < error[i + 1]:
                count += 1
            else:
                count = 0
            if count == 10:
                break


    def confusion_matrix(self, testset, testtargets):
        confusionmat = np.zeros((testset.shape[1], testset.shape[1]))		
        for n in range(testset.shape[0]):		
            pred = np.argmax(testset[n])
            true = np.argmax(testtargets[n]) 
            confusionmat[pred, true] += 1
        return confusionmat



