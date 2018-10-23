
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
        self.eta = 0.01
        self.beta = 1.0
        self.nvectors = inputs.shape[0]
        self.ntargets = targets.shape[1]
        self.ninputs = inputs.shape[1]
        self.hiddenacc = np.zeros((self.nvectors, self.nhidden))
        self.output = np.zeros((self.nvectors, targets.shape[1]))
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
        numb_vec = inputs.shape[0]
        inputs_tot = self.include_bias(inputs)
        #print(inputs_tot.shape, self.v.shape)
        h_chi = np.zeros((numb_vec, self.nhidden))
        a_chi = np.zeros((numb_vec, self.nhidden))
        #a_chi_tot = self.include_biasw(inputs)
        h_kappa = np.zeros((numb_vec, self.ntargets))
        y_kappa = np.zeros((numb_vec, self.ntargets))

        for n in range(numb_vec):
            for j in range(self.nhidden):
                for i in range(inputs.shape[1] + 1):
                    h_chi[n, j] += inputs_tot[n, i] * self.v[i, j] #With bias
                a_chi[n, j] = self.sigmoid(h_chi[n, j])
                self.hiddenacc[n, j] = a_chi[n, j]                 #no bias
        a_chi_tot = self.include_bias(a_chi)                       #With bias

        for n in range(numb_vec):        
            for i in range(self.ntargets):
                for j in range(self.nhidden + 1):
                    h_kappa[n, i] += a_chi_tot[n, j] * self.w[j, i] #With bias
                y_kappa[n,i] = self.sigmoid(h_kappa[n, i])
                self.output[n, i] = y_kappa[n,i]
        return y_kappa
    def deltaO(self, inputs, targets):
        numb_vectors = inputs.shape[0]
        delO = np.zeros((numb_vectors, self.ntargets))
        for n in range(numb_vectors):
            for j in range(self.ntargets):
                delO[n, j] = (self.output[n, j] - targets[n, j])*self.sigmoid_derivative(self.output[n, j])
        return delO

    def deltaH(self, inputs, targets): 
        numb_vectors = inputs.shape[0]
        deltaOut = self.deltaO(inputs, targets)
        delH = np.zeros((numb_vectors, self.nhidden))
        w_T = self.transpose(self.w)
        for n in range(numb_vectors):
            for k in range(self.nhidden):
                delH[n, k] = self.hiddenacc[n, k]*(1.0 - self.hiddenacc[n, k])*(sum(deltaOut[n, :]*w_T[:, k]))
        return delH

    def backward(self, inputs, targets):
        numb_vectors = inputs.shape[0]
        delH = self.deltaH(inputs, targets)
        delO = self.deltaO(inputs, targets)

        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))

        inputs_T = self.transpose(inputs)
        for i in range(inputs.shape[1]):
            for k in range(self.nhidden):
                for n in range(numb_vectors):
                    updateV[i, k] = self.eta*(inputs_T[i, n]*delH[n, k])

        hiddenacc_T = self.transpose(self.hiddenacc)
        for j in range(self.nhidden):
            for k in range(self.ntargets):
                for n in range(numb_vectors):
                    updateW[j, k] = self.eta*(hiddenacc_T[j, n]*delO[n, k])

        #updateV and updateW are one smaller than self.v and self.w because of bias 
        self.v -= updateV
        self.w -= updateW

    def Recall(self, data):
        output = self.forward(data)
        return output
    
    def train(self, inputs, targets):

        self.forward(inputs)
        self.backward(inputs, targets)

    def error(self, validationset, validationstargets):
        error = np.zeros(validationset.shape[0])
        validation_out = self.Recall(validationset)
        for i in range(validationset.shape[0]):
            error[i] = np.linalg.norm(validation_out[i] -  validationstargets[i])**2
        return error

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        epochs = 200
        error = self.error(validationset, validationstargets)
        count = 0
        for i in range(epochs - 1):
            self.train(inputs, targets)
            if error[i] < error[i +1]:
                count += 1
            else:
                count = 0
            if count == 10:
                break


