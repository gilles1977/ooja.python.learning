import numpy as np

class Network:
    def __init__(self, shape, transfer, threshold):
        self.nlayers = shape.size
        self.layers = []
        self.o = 0.
        self.e = 0.
        self.threshold = threshold
        self.lf = 0.5
        for i in range(1, self.nlayers):
            self.layers.append(Layer(shape[i], shape[i - 1], transfer))

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.o = x
        return self.threshold(self.o)

    def backprop(self, t):
        self.e = t - self.o
        self.layers[self.nlayers - 2].e = self.e
        for l in range(self.nlayers - 2, 0, -1):
            self.layers[l - 1].error(self.layers[l])
        for l in self.layers:
            l.w += ((l.x.T * l.dx) * l.e) * self.lf
            l.b += (l.dx * l.e) * self.lf

class Layer:
    def __init__(self, nneurons, ninput, transfer):
        self.w = np.random.rand(ninput, nneurons) if nneurons > 1 else np.random.rand(ninput)
        self.b = 0.1
        self.nneurons = nneurons
        self.transfer = transfer
        self.e = []
        self.x = []
        self.dx = []

    def forward(self, x):
        self.x = x
        for i in range(1, self.nneurons):
            self.x = np.vstack((self.x, x))
        a = np.dot(self.x, self.w)
        a = a[0] if a.ndim > 1 else a
        a += self.b
        self.dx = self.transfer.der(a)
        return self.transfer.out(a)

    def error(self, layer):
        self.e = np.dot(layer.w, layer.e)

    def display(self):
        print("x={0}".format(self.x))
        print("w={0}".format(self.w))
        print("b={0}".format(self.b))
        print("dx={0}".format(self.dx))
        print("e={0}".format(self.e))

class Step:
    def out(self, n):
        return np.heaviside(n, 0.5)

    def der(self, n):
        return np.ones_like(n)

class Sigmoid:
    def out(self, n):
        s = 1 / (1 + np.exp(-n)) 
        return s

    def der(self, n):
        s = self.out(n)
        return s * (1 - s)