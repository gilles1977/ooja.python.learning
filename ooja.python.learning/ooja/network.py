import numpy as np
from matplotlib import pylab
import pylab as plt
import netdraw as nd

class Network:
    def __init__(self, shape, transfer, threshold):
        self.shape = shape
        self.nlayers = shape.size
        self.layers = []
        self.o = 0.
        self.e = 0.
        self.threshold = threshold
        self.transfer = transfer
        self.lf = 0.5
        for i in range(1, self.nlayers):
            self.layers.append(Layer(shape[i], shape[i - 1], transfer))

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.o = x
        return self.threshold(self.o)

    def backprop(self, t, e):
        lf = self.transfer.out(np.abs(e))
        self.e = t - self.o
        outl = self.layers[self.nlayers - 2]
        outl.e = self.e
        outl.dx = self.transfer.der(self.o)
        for l in range(self.nlayers - 2, 0, -1):
            self.layers[l - 1].loss(self.layers[l])
            #a = np.dot(self.layers[l - 1].e, self.layers[l - 1].w)
            #self.layers[l - 1].dx = self.transfer.der(self.layers[l - 1].e)
        for l in self.layers:
            l.dx = self.transfer.der(l.a)
            l.w += (((l.x.T * l.dx) * l.e) * lf)
            #l.b += (l.dx * l.e) * lf

    def display(self, e, i):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        w = []
        for l in range(0, self.shape.size-1):
            w.append(self.layers[l].w)
        nd.draw_neural_net(ax, .1, .9, .1, .9, self.shape, w, i, e)
        for l in self.layers:
            l.display()
        plt.show()

class Layer:
    def __init__(self, nneurons, ninput, transfer):
        self.w = 2 * (np.random.rand(ninput + 1, nneurons) if nneurons > 1 else np.random.rand(ninput + 1)) - 1
        #self.b = np.ones(1)
        self.nneurons = nneurons
        self.transfer = transfer
        self.e = []
        self.x = []
        self.dx = []
        self.a = []

    def forward(self, x):
        x = np.append(x, 1)
        self.x = x
        for i in range(1, self.nneurons):
            self.x = np.vstack((self.x, x))
        a = np.dot(self.x, self.w)
        self.a = a[0] if a.ndim > 1 else a
        #a += self.b
        #self.dx = self.transfer.der(a)
        return self.transfer.out(self.a)

    def loss(self, layer):
        self.e = np.dot(layer.w[:-1], layer.e)

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
        #n = np.clip(n, -500, 500)
        s = 1 / (1 + np.exp(-n)) 
        return s

    def der(self, n):
        s = self.out(n)
        return s * (1 - s)