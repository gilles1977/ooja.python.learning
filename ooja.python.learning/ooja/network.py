import numpy as np
from matplotlib import pylab
import pylab as plt
import netdraw as nd
from sklearn import metrics
from sklearn.model_selection import train_test_split

class Network:
    def __init__(self, threshold, cost=None):
        self.nlayers = 0
        self.layers = []
        self.y = 0.
        self.E = 0.
        self.threshold = threshold
        self.cost = cost

    def addlayer(self, nneurons, ninput=None, transfer=None, cost=None):
        if ninput == None:
            ninput = self.layers[-1].nneurons
        self.layers.append(Layer(ninput, nneurons, transfer, cost))
        self.nlayers += 1

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.y = x
        return self.threshold(self.y)

    def backprop(self, t, lr):
        momentum = .9
        outlayer = self.layers[-1]

        #self.E = np.sum(np.square(t - self.y)) / 2
        self.E = outlayer.cost.out(self.y, t)
        
        analytic_dEdx = self.y - t
        
        #outlayer.dEdy = t - self.y
        #outlayer.dEdx = outlayer.transfer.der(outlayer.ti) * outlayer.dEdy
        
        outlayer.dEdy = outlayer.cost.der(self.y, t)
        outlayer.dEdx = outlayer.dEdy @ outlayer.transfer.der(outlayer.y)
        outlayer.dEdw = np.dot(outlayer.x.T,  outlayer.dEdx)
        outlayer.dEdb = np.sum(outlayer.dEdx, axis=0)

        for l in range(self.nlayers - 1, 0, -1):
            self.layers[l - 1].gradient(self.layers[l])

        for l in self.layers:
            l.w -= (lr * l.dEdw)
            l.b -= (lr * l.dEdb)

    def display(self, e, i):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        w = []
        b = []
        shape = [self.layers[0].x.size]
        for l in range(0, self.nlayers):
            w.append(self.layers[l].w)
            b.append(self.layers[l].b)
            shape.append(self.layers[l].nneurons)
        nd.draw_neural_net(ax, .1, .9, .1, .9, shape, w, b, i, e)
        for l in self.layers:
            l.display()
        plt.show()

    def sgd(self, X, y):
        total_error = 0.
        for j in range(0, len(X)):
            output = self.forward(X[j])
            self.backprop(y[j], .01)
            total_error += self.E
        total_error = total_error / len(X)
        return total_error

    def minibatch(self, X, y, minibatchsize):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        total_error = 0.
        for j in range(0, X_train.shape[0], minibatchsize):
            X_train_mini = X_train[j:j + minibatchsize]
            y_train_mini = y_train[j:j + minibatchsize]

            output = self.forward(X_train_mini)
            self.backprop(y_train_mini, .01)
            total_error += self.E

        total_error = total_error / len(X_train)
        return total_error

class Layer:
    def __init__(self, ninput, nneurons, transfer, cost):
        self.w = 2 * (np.random.rand(ninput, nneurons)) - 1
        self.b = 2 * np.random.rand(nneurons) - 1
        self.nneurons = nneurons
        self.transfer = transfer
        self.cost = cost
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
        print("x={0}".format(self.x.shape))
        print("w={0}".format(self.w.shape))
        print("dEdy={0}".format(self.dEdy))
        print("dEdx={0}".format(self.dEdx))
        print("dEdw={0}".format(self.dEdw))
        print("dEdb={0}".format(self.dEdb))

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

    def jacobian(self, n):
        s = n.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def der(self, n):
        #return self.jacobian(n)
        res = np.array([self.jacobian(row) for row in n])
        return res[0]

class Cost:
    class CrossEntropy:
        def out(self, o, t, epsilon=1e-12):
            o = np.clip(o, epsilon, 1. - epsilon)
            N = o.shape[0]
            return -np.sum(t * np.log(o + 1e-9))/N

        def der(self, o, t):
            return -t / o

    class MeanSquareError:
        def out(self, o, t):
            return np.sum(np.square(t - o)) / 2

        def der(self, o, t):
            return t - o

class Error:
    def out(self, y, t):
        return ((y - t) ** 2) / 2

    def der(self, y, t):
        return y - t