import numpy as np

class Network:
    def __init__(self, shape, transfer):
        self.nlayers = shape.size
        self.layers = []
        self.o = []
        for i in range(1, self.nlayers):
            self.layers.append(Layer(shape[i], shape[i - 1], transfer))

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.o = x
        return self.o

    def backprop(self, t):
        self.layers[self.nlayers - 2].e = t - self.o
        for l in range(self.nlayers - 2, 0, -1):
            self.layers[l - 1].error(self.layers[l])
        for l in self.layers:
            l.w += l.e * l.dx * l.x

class Layer:
    def __init__(self, nneurons, ninput, transfer):
        self.w = np.random.rand(ninput, nneurons)
        self.b = np.ones(nneurons) * 0.1
        self.transfer = transfer
        self.e = []
        self.x = []
        self.dx = []

    def forward(self, x):
        self.x = x
        a = np.dot(self.x, self.w) + self.b
        self.dx = self.transfer.der(a)
        return self.transfer.out(a)

    def error(self, layer):
        self.e = np.dot(layer.w, layer.e)

    def display(self):
        print("x={0}".format(self.x))
        print("w={0}".format(self.w))
        print("b={0}".format(self.b))

class Node:
    """a node in the network"""
    def __init__(self, id, transfer, bias, is_output=False):
        self.id = id
        self.transfer = transfer
        self.bias = bias
        self.x = []
        self.w = []

    def activation(self):
        return self.bias + dot(self.input.w, self.input.x)
    
    def output(self):
        return self.transfer.output(self.activation())

class Link:
    """a link between nodes"""
    def __init__(self):
        self.source = []
        self.destination = []
        self.x = []
        self.w = []

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