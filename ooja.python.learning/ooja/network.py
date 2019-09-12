import numpy as np
from matplotlib import pylab
import pylab as plt
import netdraw as nd
from sklearn import metrics

class Network:
    def __init__(self, threshold):
        self.nlayers = 0
        self.layers = []
        self.y = 0.
        self.E = 0.
        self.threshold = threshold

    def addlayer(self, nneurons, ninput=None, transfer=None):
        if ninput == None:
            ninput = self.layers[-1].nneurons
        self.layers.append(Layer(ninput, nneurons, transfer))
        self.nlayers += 1

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.y = x
        return self.threshold(self.y)

    def backprop(self, t, lr):
        momentum = .9
        outlayer = self.layers[-1]

        self.E = np.sum(np.square(t - self.y)) / 2

        outlayer.dEdy = t - self.y
        outlayer.dEdx = outlayer.transfer.der(outlayer.ti) * outlayer.dEdy
        outlayer.dEdw = np.dot(outlayer.x.T,  outlayer.dEdx)
        outlayer.dEdb = np.sum(outlayer.dEdx, axis=0)

        for l in range(self.nlayers - 1, 0, -1):
            self.layers[l - 1].gradient(self.layers[l])

        for l in self.layers:
            l.w += (lr * l.dEdw)
            l.b += (lr * l.dEdb)

    def display(self, e, i):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        w = []
        b = []
        for l in range(0, self.nlayers):
            w.append(self.layers[l].w)
            b.append(self.layers[l].b)
        #nd.draw_neural_net(ax, .1, .9, .1, .9, self.shape, w, b, i, e)
        for l in self.layers:
            l.display()
        #plt.show()

class Layer:
    def __init__(self, ninput, nneurons, transfer):
        self.w = 2 * (np.random.rand(ninput, nneurons)) - 1
        self.b = 2 * np.random.rand(nneurons) - 1
        self.nneurons = nneurons
        self.transfer = transfer
        self.e = None
        self.x = None
        self.y = None
        self.ti = None
        self.dEdx = None
        self.dEdy = None
        self.dEdw = None
        self.dEdb = None

    def forward(self, x):
        self.x = np.atleast_2d(x)
        self.ti = np.dot(self.x, self.w) + self.b
        self.y = self.transfer.out(self.ti)
        return self.y

    def gradient(self, layer):
        self.dEdy = np.dot(layer.dEdx, layer.w.T)
        self.dEdx = self.transfer.der(self.ti) * self.dEdy
        self.dEdw = np.dot(self.x.T,  self.dEdx)
        self.dEdb = np.sum(self.dEdx, axis=0)

    def display(self):
        print("x={0}".format(self.x))
        print("w={0}".format(self.w))
        print("dx={0}".format(self.dx))
        print("e={0}".format(self.e))

class Step:
    def out(self, n):
        return np.heaviside(n, 0.5)

    def der(self, n):
        return np.ones_like(n)

class Sigmoid:
    def out(self, n):
        n = np.clip(n, -20, 20)
        s = 1 / (1 + np.exp(-n)) 
        return s

    def der(self, n):
        s = self.out(n)
        return s * (1 - s)

class Relu:
    def out(self, n):
        n = np.clip(n, -20, 20)
        return np.maximum(0.01 * n, n)
    
    def der(self, n):
        gradients = 1. * (n > 0.01)
        gradients[gradients == 0] = 0.01
        return gradients

class Softmax:
    def out(self, n):
        exps = np.exp(n - n.max())
        return exps / np.sum(exps)

    def der(self, n):
        jacobian = np.diag(n)
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:
                    jacobian[i][j] = n[i] * (1 - n[j])
                else:
                    jacobian[i][j] = -n[i] * n[j]
        return jacobian

class Loss:
    def CrossEntropy(self, o, t):
        return metrics.log_loss(t, o)

    def MeanSquareError(self, o, t):
        return np.square(t - o).mean()

class Error:
    def out(self, y, t):
        return ((y - t) ** 2) / 2

    def der(self, y, t):
        return y - t