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
        self.lf = 0.5
        for i in range(1, self.nlayers + 1):
            self.layers.append(Layer(shape[i], shape[i - 1], transfer))

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.o = x
        return self.threshold(self.o)

    def backprop(self, t, e):
        lf = self.transfer.out(np.abs(e))
        self.e = t - self.o
        outl = self.layers[self.nlayers - 1]
        outl.e = self.e
        outl.dx = self.e * self.transfer.der(self.o)
        for l in range(self.nlayers - 1, 0, -1):
            self.layers[l - 1].loss(self.layers[l])
        for l in self.layers:
            l.w += (l.dx * np.atleast_2d(l.x).T * lf).T
            l.b += l.dx * lf

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
        plt.show()

class Layer:
    def __init__(self, nneurons, ninput, transfer):
        self.w = 2 * (np.random.rand(nneurons, ninput) if nneurons > 1 else np.random.rand(ninput)) - 1
        self.b = np.random.rand(nneurons)
        self.nneurons = nneurons
        self.transfer = transfer
        self.e = []
        self.x = []
        self.y = []
        self.dx = []

    def forward(self, x):
        self.x = x
        self.y = self.transfer.out(np.dot(self.w, self.x) + self.b)
        return self.y

    def loss(self, layer):
        self.e = np.dot(layer.dx, layer.w)
        self.dx = self.e * self.transfer.der(self.y)

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