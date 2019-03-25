import numpy as np
import math


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    '''dReLU(x) = {0 if x < 0; 1 if x > 0}.'''
    return ((x/np.abs(x)) + 1) / 2


def LeakyReLU(x, a=0.02):
    return np.maximum(a*x, x)


def dLeakyReLU(x, a=0.02):
    '''dReLU(x) = {a if x < 0; 1 if x > 0}.'''
    return ((x/np.abs(x) + 1) / 2) * (1-a) + a


def Sigmoid(x):
    '''Sigmoid(x) = 1/(1+exp(-x)).'''
    import scipy.special
    return scipy.special.expit(x)


def dSigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


class DNN:

    def __init__(self, layout, actFunc, dactFunc):
        '''Initializing the Deep Neural Network Parameters.'''
        self.w = {}  # weights of the nodes
        self.b = {}  # biases of the nodes
        self.l = len(layout) - 1  # depth of the network
        self.actFunc = actFunc
        self.dactFunc = dactFunc
        for i in range(1, len(layout)):
            # Initializing the weights and biases randomly
            self.w[i] = np.random.randn(layout[i], layout[i-1])
            self.b[i] = np.random.randn(layout[i], 1)

    def queryR(self, x, i=-1):
        '''Querying the Deep Neural Network with a recursive function.'''
        if i == -1:
            i = self.l
        elif i == 0:
            return x
        return self.actFunc[i-1](
            np.dot(self.w[i], self.queryR(x, (i-1))) + self.b[i])

    def queryI(self, x):
        '''Querying the Deep Neural Network with an iterative function.'''
        a = x
        for i in range(self.l):
            a = self.actFunc[i](np.dot(self.w[i+1], a) + self.b[i+1])
        return a

    def train(self, inputs, targets, lr=0.1,
              epoch=1, mbs=256, beta1=0.9, beta2=0.99):

        A, Z, dA, dZ, Vdw, Vdb, Sdw, Sdb = {}, {}, {}, {}, {}, {}, {}, {}
        mbn = math.ceil(inputs.shape[1] / mbs)  # Number of Mini-Batches
        e = 10**(-6)  # epsilon
        for i in range(self.l):
            Vdw[i+1] = Vdb[i+1] = Sdw[i+1] = Sdb[i+1] = 0
        for j in range(epoch):
            for k in range(mbn):
                start = mbs*i
                end = mbs*(i+1)
                if end > inputs.shape[1]:
                    end = inputs.shape[1]
                X = inputs[:, start:end]
                Y = targets[:, start:end]
                A[0] = X
                m = X.shape[1]
                for i in range(self.l):
                    # Forward Propagation step
                    Z[i+1] = np.dot(self.w[i+1], A[i]) + self.b[i+1]
                    A[i+1] = self.actFunc[i](Z[i+1])

                dA[self.l] = -Y/A[self.l] + (1-Y)/(1-A[self.l])
                for i in range(self.l):
                    # Backward Propagation step with ADAM optimization
                    dZ[self.l-i] = dA[self.l-i]*self.dactFunc[self.l-i-1](
                        Z[self.l-i])

                    Vdw[self.l-i] = (beta1*Vdw[self.l-i] + (1-beta1)*np.dot(
                        dZ[self.l-i], A[self.l-i-1].T))
                    Sdw[self.l-i] = (beta2*Sdw[self.l-i] + (1-beta2)*np.dot(
                        dZ[self.l-i], A[self.l-i-1].T)**2)
                    self.w[self.l-i] = self.w[self.l-i] - \
                        (lr/m)*Vdw[self.l-i]/(np.sqrt(Sdw[self.l-i]) + e)

                    Vdb[self.l-i] = (beta1*Vdb[self.l-i] + (1-beta1)*np.sum(
                        dZ[self.l-i], axis=1, keepdims=True))
                    Sdb[self.l-i] = (beta2*Sdb[self.l-i] + (1-beta2)*np.sum(
                        dZ[self.l-i], axis=1, keepdims=True)**2)
                    self.b[self.l-i] = self.b[self.l-i] - \
                        (lr/m)*Vdb[self.l-i]/(np.sqrt(Sdb[self.l-i]) + e)

                    dA[self.l-i-1] = np.dot(self.w[self.l-i].T, dZ[self.l-i])


# load the training set
print('Loading the training set...')
data = np.load('./mnist_dataset/mnist_data_numpy.npz')

inputs = data['inputs']
targets = data['targets']

# train the network
# create an instance of the deep neural network
n2 = DNN([28*28, 100, 100, 10],
         [Sigmoid, Sigmoid, Sigmoid],
         [dSigmoid, dSigmoid, dSigmoid])

lr = 10
epoch = 10
mbs = 512
print('--------------------------------------')
print('Training initiated\nlr =', lr, 'epoch =', epoch, '\
mbs =', mbs)
n2.train(inputs, targets, lr=lr, epoch=epoch, mbs=mbs)

print('Training completed.\nTesting the model using a test set...')

# Testing the network on a test set
with open('./mnist_dataset/mnist_test.csv') as test_data_file:
    test_data_list = test_data_file.readlines()
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = 0.99*np.asfarray(all_values[1:]).reshape(28*28, 1)/255.0+0.01
    outputs = n2.queryI(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

print('Accuracy =', 100*sum(scorecard)/len(scorecard), '%')
