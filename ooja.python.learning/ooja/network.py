import numpy as np
from matplotlib import pylab
import pylab as plt
import netdraw as nd

class Network:
    def __init__(self, shape, transfer, threshold):
        self.shape = shape
        self.nlayers = shape.size - 1
        self.layers = []
        self.o = 0.
        self.e = 0.
        self.threshold = threshold
        self.transfer = transfer
        for i in range(1, self.nlayers + 1):
            self.layers.append(Layer(shape[i - 1], shape[i], transfer))

    def addlayer(self, ninput, nneurons, transfer):
        self.layers.append(Layer(ninput, nneurons, transfer))
        self.nlayers += 1

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.o = x
        return self.threshold(self.o)

    def backprop(self, t, e):
        lr = self.transfer.out(np.abs(e))
        alpha = 1. -lr
        self.e = t - self.o
        self.layers[-1].e = self.e
        self.layers[-1].dx = self.e * self.layers[-1].transfer.der(self.o)
        for l in range(self.nlayers - 1, 0, -1):
            self.layers[l - 1].loss(self.layers[l])
        for l in self.layers:
            l.dw = (lr * np.atleast_2d(l.x).T * l.dx) + (alpha * l.dw)
            l.w += l.dw
            l.db = (lr * l.dx) + (alpha * l.db)
            l.b += l.db

    def display(self, e, i):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        w = []
        b = []
        for l in range(0, self.shape.size-1):
            w.append(self.layers[l].w)
            b.append(self.layers[l].b)
        nd.draw_neural_net(ax, .1, .9, .1, .9, self.shape, w, b, i, e)
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
        self.o = None
        self.dx = None
        self.dw = 0.
        self.db = 0.

    def forward(self, x):
        self.x = x
        self.o = self.transfer.out(np.dot(self.x, self.w) + self.b)
        return self.o

    def loss(self, layer):
        self.e = np.dot(layer.dx, layer.w.T)
        self.dx = self.e * self.transfer.der(self.o)

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