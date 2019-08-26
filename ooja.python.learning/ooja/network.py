import numpy as np

class Network:
    def __init__(self, shape, ninput, transfer, threshold):
        self.nlayers = shape.size

class Layer:
    def __init__(self, nneurons, ninput):
        self.w = np.random.rand(ninput, nneurons)
        self.b = np.ones(nneurons) * 0.1

    def forward(self, x):
        return np.dot(x, self.w)

    def backward(self, x, d):


class Node:
    """a node in the network"""
    def __init__(self, id, transfer, bias, is_output = False):
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
